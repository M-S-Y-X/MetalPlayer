//
//  MetalViewController.m
//  完整版：全图超分、拼接 + 局部超分（修正模式切换）
//

#import "MetalViewController.h"
#import "SREngine.h"
#import "ImageStitch.h"
#import "StitchTypes.h"
#import "HomographyEngine.h"

#define YUV_WIDTH       1280
#define YUV_HEIGHT      1280
#define PLAYBACK_FPS    30
#define YUV_FILE_PATH   [[NSBundle mainBundle] pathForResource:@"1280_1280" ofType:@"yuv"]

#define RIGHT_YUV_WIDTH       2880
#define RIGHT_YUV_HEIGHT      1620
#define RIGHT_YUV_FILE_PATH   [[NSBundle mainBundle] pathForResource:@"2880_1620_L" ofType:@"yuv"]
#define LEFT_YUV_WIDTH       2880
#define LEFT_YUV_HEIGHT      1620
#define LEFT_YUV_FILE_PATH   [[NSBundle mainBundle] pathForResource:@"2880_1620_R" ofType:@"yuv"]

typedef struct {
    vector_float2 position;
    vector_float2 texCoord;
} Vertex;

typedef NS_ENUM(NSInteger, DisplayMode) {
    DisplayModeOff = 0,
    DisplayModeLanczos,
    DisplayModeIBP,
    DisplayModeTemporalIBP,
    DisplayModeTemporalPlus,
    DisplayModeDeepLearning,
    DisplayModeStitch
};

@interface MetalViewController () <MTKViewDelegate>
@property (nonatomic, strong) MTKView *metalView;
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLRenderPipelineState> yuvPipelineState;
@property (nonatomic, strong) id<MTLRenderPipelineState> rgbaPipelineState;
@property (nonatomic, strong) id<MTLBuffer> vertexBuffer;

@property (nonatomic, strong) id<MTLTexture> yTexture;
@property (nonatomic, strong) id<MTLTexture> uTexture;
@property (nonatomic, strong) id<MTLTexture> vTexture;
@property (nonatomic, strong) NSData *fullYUVData;
@property (nonatomic, assign) NSInteger frameSize;
@property (nonatomic, assign) NSInteger totalFrames;
@property (nonatomic, assign) NSInteger currentFrame;
@property (nonatomic, strong) NSTimer *playbackTimer;

@property (nonatomic, strong) SREngine *srEngine;

@property (nonatomic, assign) float customScale;
@property (nonatomic, strong) NSTextField *scaleTextField;
@property (nonatomic, strong) NSButton *modeButton;

@property (nonatomic, strong) ImageStitcher *stitcher;
@property (nonatomic, strong) id<MTLTexture> leftRGBA;
@property (nonatomic, strong) id<MTLTexture> rightRGBA;
@property (nonatomic, strong) id<MTLTexture> stitchResult;
@property (nonatomic, assign) DisplayMode currentMode;
@property (nonatomic, assign) BOOL stitchReady;

@property (nonatomic, assign) CGImageRef leftCGImage;
@property (nonatomic, assign) CGImageRef rightCGImage;
@property (nonatomic, strong) NSData *leftImageData;
@property (nonatomic, strong) NSData *rightImageData;

// ---------- 局部超分相关 ----------
@property (nonatomic, assign) BOOL localZoomMode;
@property (nonatomic, assign) CGRect selectedRect;           // 像素坐标
@property (nonatomic, strong) NSTextField *xField;
@property (nonatomic, strong) NSTextField *yField;
@property (nonatomic, strong) NSTextField *wField;
@property (nonatomic, strong) NSTextField *hField;
@property (nonatomic, strong) NSButton *localZoomBtn;
@property (nonatomic, strong) NSButton *exitLocalBtn;
@end

@implementation MetalViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.device = MTLCreateSystemDefaultDevice();
    if (!self.device) { NSLog(@"❌ Metal not supported"); return; }

    self.metalView = [[MTKView alloc] initWithFrame:self.view.bounds device:self.device];
    self.metalView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.metalView.delegate = self;
    self.metalView.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    self.metalView.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    [self.view addSubview:self.metalView];

    self.commandQueue = [self.device newCommandQueue];
    self.customScale = 0.0;
    self.currentMode = DisplayModeOff;

    self.srEngine = [[SREngine alloc] initWithDevice:self.device];
    self.stitcher = [[ImageStitcher alloc] initWithDevice:self.device];

    [self setupRenderPipelines];
    [self setupVertexBuffer];
    [self loadYUVFileAndPrepareTextures];
    [self loadStitchImages];
    [self setupUI];
}

- (void)dealloc {
    [self.playbackTimer invalidate];
    if (_leftCGImage) CGImageRelease(_leftCGImage);
    if (_rightCGImage) CGImageRelease(_rightCGImage);
}

// ---------- UI 布局 ----------
- (void)setupUI {
    CGFloat yPos = self.view.bounds.size.height - 50;

    // 模式切换按钮
    self.modeButton = [[NSButton alloc] initWithFrame:NSMakeRect(20, yPos, 200, 30)];
    self.modeButton.title = @"Mode: Off (Bilinear)";
    self.modeButton.bezelStyle = NSBezelStyleRounded;
    self.modeButton.wantsLayer = YES;
    self.modeButton.layer.backgroundColor = [[NSColor colorWithRed:0.2 green:0.6 blue:0.2 alpha:0.8] CGColor];
    [self.modeButton setAttributedTitle:[[NSAttributedString alloc] initWithString:self.modeButton.title
                                                                       attributes:@{NSForegroundColorAttributeName: [NSColor whiteColor]}]];
    self.modeButton.autoresizingMask = NSViewMinYMargin | NSViewMaxXMargin;
    self.modeButton.target = self;
    self.modeButton.action = @selector(cycleMode:);
    [self.view addSubview:self.modeButton positioned:NSWindowAbove relativeTo:self.metalView];

    yPos -= 40;
    // 缩放输入
    NSTextField *label = [[NSTextField alloc] initWithFrame:NSMakeRect(20, yPos, 60, 22)];
    [label setStringValue:@"Scale:"];
    [label setBezeled:NO];
    [label setDrawsBackground:NO];
    [label setEditable:NO];
    [label setSelectable:NO];
    label.autoresizingMask = NSViewMinYMargin | NSViewMaxXMargin;
    [self.view addSubview:label];

    self.scaleTextField = [[NSTextField alloc] initWithFrame:NSMakeRect(80, yPos, 60, 22)];
    self.scaleTextField.stringValue = @"0";
    self.scaleTextField.placeholderString = @"e.g. 2.0";
    self.scaleTextField.autoresizingMask = NSViewMinYMargin | NSViewMaxXMargin;
    [self.view addSubview:self.scaleTextField];

    NSButton *applyBtn = [[NSButton alloc] initWithFrame:NSMakeRect(150, yPos, 80, 28)];
    applyBtn.title = @"Apply";
    applyBtn.bezelStyle = NSBezelStyleRounded;
    applyBtn.wantsLayer = YES;
    applyBtn.layer.backgroundColor = [[NSColor orangeColor] CGColor];
    [applyBtn setAttributedTitle:[[NSAttributedString alloc] initWithString:@"Apply"
                                                                 attributes:@{NSForegroundColorAttributeName: [NSColor whiteColor]}]];
    applyBtn.autoresizingMask = NSViewMinYMargin | NSViewMaxXMargin;
    applyBtn.target = self;
    applyBtn.action = @selector(applyCustomScale:);
    [self.view addSubview:applyBtn positioned:NSWindowAbove relativeTo:self.metalView];

    // ---------- 局部超分 UI ----------
    yPos -= 40;
    NSTextField *roiLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(20, yPos, 60, 22)];
    [roiLabel setStringValue:@"ROI:"];
    [roiLabel setBezeled:NO];
    [roiLabel setDrawsBackground:NO];
    [roiLabel setEditable:NO];
    [roiLabel setSelectable:NO];
    roiLabel.autoresizingMask = NSViewMinYMargin | NSViewMaxXMargin;
    [self.view addSubview:roiLabel];

    CGFloat fieldW = 50, fieldH = 22, gap = 5;
    CGFloat startX = 80;
    self.xField = [self makeTextField:CGRectMake(startX, yPos, fieldW, fieldH) placeholder:@"x"];
    startX += fieldW + gap;
    self.yField = [self makeTextField:CGRectMake(startX, yPos, fieldW, fieldH) placeholder:@"y"];
    startX += fieldW + gap;
    self.wField = [self makeTextField:CGRectMake(startX, yPos, fieldW, fieldH) placeholder:@"w"];
    startX += fieldW + gap;
    self.hField = [self makeTextField:CGRectMake(startX, yPos, fieldW, fieldH) placeholder:@"h"];

    self.localZoomBtn = [[NSButton alloc] initWithFrame:NSMakeRect(startX + fieldW + 10, yPos, 90, 28)];
    self.localZoomBtn.title = @"局部超分";
    self.localZoomBtn.bezelStyle = NSBezelStyleRounded;
    self.localZoomBtn.wantsLayer = YES;
    self.localZoomBtn.layer.backgroundColor = [[NSColor colorWithRed:0.2 green:0.6 blue:0.8 alpha:0.8] CGColor];
    [self.localZoomBtn setAttributedTitle:[[NSAttributedString alloc] initWithString:@"局部超分"
                                                                         attributes:@{NSForegroundColorAttributeName: [NSColor whiteColor]}]];
    self.localZoomBtn.autoresizingMask = NSViewMinYMargin | NSViewMaxXMargin;
    self.localZoomBtn.target = self;
    self.localZoomBtn.action = @selector(enterLocalZoom:);
    [self.view addSubview:self.localZoomBtn positioned:NSWindowAbove relativeTo:self.metalView];

    self.exitLocalBtn = [[NSButton alloc] initWithFrame:NSMakeRect(self.localZoomBtn.frame.origin.x + 100, yPos, 90, 28)];
    self.exitLocalBtn.title = @"退出局部";
    self.exitLocalBtn.bezelStyle = NSBezelStyleRounded;
    self.exitLocalBtn.wantsLayer = YES;
    self.exitLocalBtn.layer.backgroundColor = [[NSColor redColor] CGColor];
    [self.exitLocalBtn setAttributedTitle:[[NSAttributedString alloc] initWithString:@"退出局部"
                                                                          attributes:@{NSForegroundColorAttributeName: [NSColor whiteColor]}]];
    self.exitLocalBtn.autoresizingMask = NSViewMinYMargin | NSViewMaxXMargin;
    self.exitLocalBtn.target = self;
    self.exitLocalBtn.action = @selector(exitLocalZoom:);
    [self.view addSubview:self.exitLocalBtn positioned:NSWindowAbove relativeTo:self.metalView];
}

- (NSTextField *)makeTextField:(CGRect)frame placeholder:(NSString *)placeholder {
    NSTextField *tf = [[NSTextField alloc] initWithFrame:frame];
    tf.stringValue = @"";
    tf.placeholderString = placeholder;
    tf.autoresizingMask = NSViewMinYMargin | NSViewMaxXMargin;
    [self.view addSubview:tf];
    return tf;
}

// ---------- 模式切换（修正版：不再自动退出局部模式，仅拼接时退出） ----------
- (void)cycleMode:(NSButton *)sender {
    NSInteger totalModes = 7;
    NSInteger newMode = (self.currentMode + 1) % totalModes;
    self.currentMode = (DisplayMode)newMode;
    NSLog(@"🔄 Mode switched to: %ld", (long)self.currentMode);

    // 只有切换到拼接模式时，才强制退出局部模式（因为拼接是全图操作）
    if (self.currentMode == DisplayModeStitch && self.localZoomMode) {
        [self exitLocalZoom:nil];
    }

    if (self.currentMode == DisplayModeStitch) {
        self.srEngine.mode = SRModeOff;
        if (!self.stitchReady) [self performStitch];
    } else {
        // 其他超分模式：直接更新引擎，即使在局部模式下也有效
        SRMode srMode = (SRMode)(self.currentMode);
        self.srEngine.mode = srMode;
        [self.srEngine resetHistory];
        self.stitchReady = NO;
        self.stitchResult = nil;
    }

    switch (self.currentMode) {
        case DisplayModeOff:            sender.title = @"Mode: Off (Bilinear)"; break;
        case DisplayModeLanczos:        sender.title = @"Mode: Lanczos only"; break;
        case DisplayModeIBP:            sender.title = @"Mode: IBP (single)"; break;
        case DisplayModeTemporalIBP:    sender.title = @"Mode: IBP+Temporal"; break;
        case DisplayModeTemporalPlus:   sender.title = @"Mode: Temporal+"; break;
        case DisplayModeDeepLearning:   sender.title = @"Mode: DeepLearning"; break;
        case DisplayModeStitch:         sender.title = @"Mode: Stitch"; break;
    }
    [self.metalView setNeedsDisplay:YES];
}

- (void)applyCustomScale:(NSButton *)sender {
    float scale = self.scaleTextField.floatValue;
    if (scale < 0) scale = 0;
    if (scale > 10.0) scale = 10.0;
    self.customScale = scale;
    [self.srEngine resetHistory];
    if (scale > 0) {
        CGFloat targetPixelW = YUV_WIDTH * scale;
        CGFloat targetPixelH = YUV_HEIGHT * scale;
        CGFloat backingScale = self.view.window.backingScaleFactor ?: [[NSScreen mainScreen] backingScaleFactor];
        NSSize contentPoints = NSMakeSize(targetPixelW / backingScale, targetPixelH / backingScale);
        NSWindow *window = self.view.window;
        NSRect oldFrame = window.frame;
        CGFloat centerX = NSMidX(oldFrame);
        CGFloat centerY = NSMidY(oldFrame);
        NSRect newContentRect = NSMakeRect(0, 0, contentPoints.width, contentPoints.height);
        NSRect newFrame = [window frameRectForContentRect:newContentRect];
        newFrame.origin.x = centerX - newFrame.size.width / 2;
        newFrame.origin.y = centerY - newFrame.size.height / 2;
        NSRect visibleRect = [window.screen visibleFrame];
        if (newFrame.size.width > visibleRect.size.width) newFrame.size.width = visibleRect.size.width;
        if (newFrame.size.height > visibleRect.size.height) newFrame.size.height = visibleRect.size.height;
        if (newFrame.origin.x < visibleRect.origin.x) newFrame.origin.x = visibleRect.origin.x;
        if (newFrame.origin.y < visibleRect.origin.y) newFrame.origin.y = visibleRect.origin.y;
        [window setFrame:newFrame display:YES animate:YES];
    }
    [self.metalView setNeedsDisplay:YES];
}

// ---------- 局部超分动作 ----------
- (void)enterLocalZoom:(NSButton *)sender {
    NSInteger x = self.xField.integerValue;
    NSInteger y = self.yField.integerValue;
    NSInteger w = self.wField.integerValue;
    NSInteger h = self.hField.integerValue;

    if (w <= 0 || h <= 0 || x < 0 || y < 0 || x + w > YUV_WIDTH || y + h > YUV_HEIGHT) {
        NSLog(@"❌ 无效 ROI: %ld,%ld,%ld,%ld", x, y, w, h);
        return;
    }
    // 对齐到 YUV420 偶数边界
    x = x / 2 * 2;
    y = y / 2 * 2;
    w = (w + 1) / 2 * 2;
    h = (h + 1) / 2 * 2;

    self.selectedRect = CGRectMake(x, y, w, h);
    self.localZoomMode = YES;
    [self.srEngine resetHistory];
    NSLog(@"🔍 进入局部超分: ROI=%@", NSStringFromRect(self.selectedRect));
    [self.metalView setNeedsDisplay:YES];
}

- (void)exitLocalZoom:(NSButton *)sender {
    self.localZoomMode = NO;
    [self.srEngine resetHistory];
    NSLog(@"🔍 退出局部超分");
    [self.metalView setNeedsDisplay:YES];
}

// ---------- Metal 管线 ----------
- (void)setupRenderPipelines {
    id<MTLLibrary> lib = [self.device newDefaultLibrary];
    id<MTLFunction> vert = [lib newFunctionWithName:@"vertex_main"];
    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vert;
    desc.colorAttachments[0].pixelFormat = self.metalView.colorPixelFormat;
    NSError *err;

    id<MTLFunction> yuvFrag = [lib newFunctionWithName:@"fragment_yuv_to_rgb"];
    if (yuvFrag) {
        desc.fragmentFunction = yuvFrag;
        self.yuvPipelineState = [self.device newRenderPipelineStateWithDescriptor:desc error:&err];
        if (!self.yuvPipelineState) NSLog(@"❌ YUV pipeline error: %@", err);
    }

    id<MTLFunction> rgbaFrag = [lib newFunctionWithName:@"fragment_rgba_display"];
    if (rgbaFrag) {
        desc.fragmentFunction = rgbaFrag;
        self.rgbaPipelineState = [self.device newRenderPipelineStateWithDescriptor:desc error:&err];
        if (!self.rgbaPipelineState) NSLog(@"❌ RGBA pipeline error: %@", err);
        else NSLog(@"✅ RGBA pipeline created");
    } else {
        NSLog(@"⚠️ fragment_rgba_display not found");
    }
}

- (void)setupVertexBuffer {
    static const Vertex vertices[] = {
        {{-1.0,  1.0}, {0.0, 0.0}},
        {{ 1.0,  1.0}, {1.0, 0.0}},
        {{-1.0, -1.0}, {0.0, 1.0}},
        {{ 1.0, -1.0}, {1.0, 1.0}},
        {{ 1.0,  1.0}, {1.0, 0.0}},
        {{-1.0, -1.0}, {0.0, 1.0}}
    };
    self.vertexBuffer = [self.device newBufferWithBytes:vertices length:sizeof(vertices) options:MTLResourceStorageModeShared];
}

#pragma mark - YUV Loading
- (void)createTextures {
    MTLTextureDescriptor *yDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:YUV_WIDTH height:YUV_HEIGHT mipmapped:NO];
    yDesc.usage = MTLTextureUsageShaderRead;
    self.yTexture = [self.device newTextureWithDescriptor:yDesc];
    MTLTextureDescriptor *uvDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:YUV_WIDTH/2 height:YUV_HEIGHT/2 mipmapped:NO];
    uvDesc.usage = MTLTextureUsageShaderRead;
    self.uTexture = [self.device newTextureWithDescriptor:uvDesc];
    self.vTexture = [self.device newTextureWithDescriptor:uvDesc];
}

- (void)loadYUVFileAndPrepareTextures {
    NSString *path = YUV_FILE_PATH;
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) { NSLog(@"❌ YUV file not found"); return; }
    NSError *err;
    self.fullYUVData = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:&err];
    if (!self.fullYUVData) { NSLog(@"❌ Read error"); return; }
    NSInteger ySize = YUV_WIDTH * YUV_HEIGHT;
    NSInteger uvSize = (YUV_WIDTH/2) * (YUV_HEIGHT/2);
    self.frameSize = ySize + 2 * uvSize;
    self.totalFrames = self.fullYUVData.length / self.frameSize;
    if (self.totalFrames == 0) return;
    [self createTextures];
    self.currentFrame = 0;
    [self updateTexturesForFrame:0];
    if (self.totalFrames > 1) {
        self.playbackTimer = [NSTimer scheduledTimerWithTimeInterval:1.0/PLAYBACK_FPS target:self selector:@selector(nextFrame) userInfo:nil repeats:YES];
    }
    [self.metalView setNeedsDisplay:YES];
}

- (void)updateTexturesForFrame:(NSInteger)frameIndex {
    if (!self.fullYUVData || frameIndex >= self.totalFrames) return;
    const uint8_t *bytes = (const uint8_t *)self.fullYUVData.bytes + frameIndex * self.frameSize;
    NSInteger ySize = YUV_WIDTH * YUV_HEIGHT;
    NSInteger uvSize = (YUV_WIDTH/2) * (YUV_HEIGHT/2);
    [self.yTexture replaceRegion:MTLRegionMake2D(0, 0, YUV_WIDTH, YUV_HEIGHT) mipmapLevel:0 withBytes:bytes bytesPerRow:YUV_WIDTH];
    [self.uTexture replaceRegion:MTLRegionMake2D(0, 0, YUV_WIDTH/2, YUV_HEIGHT/2) mipmapLevel:0 withBytes:bytes + ySize + uvSize bytesPerRow:YUV_WIDTH/2];
    [self.vTexture replaceRegion:MTLRegionMake2D(0, 0, YUV_WIDTH/2, YUV_HEIGHT/2) mipmapLevel:0 withBytes:bytes + ySize bytesPerRow:YUV_WIDTH/2];
}

- (void)nextFrame {
    self.currentFrame = (self.currentFrame + 1) % self.totalFrames;
    [self updateTexturesForFrame:self.currentFrame];
    dispatch_async(dispatch_get_main_queue(), ^{ [self.metalView setNeedsDisplay:YES]; });
}

#pragma mark - Stitch Images
- (void)loadStitchImages {
    NSString *leftPath = LEFT_YUV_FILE_PATH, *rightPath = RIGHT_YUV_FILE_PATH;
    if (!leftPath || !rightPath) return;
    if (![[NSFileManager defaultManager] fileExistsAtPath:leftPath] || ![[NSFileManager defaultManager] fileExistsAtPath:rightPath]) {
        NSLog(@"❌ Stitch YUV files missing");
        return;
    }
    NSData *leftRaw = [NSData dataWithContentsOfFile:leftPath];
    NSData *rightRaw = [NSData dataWithContentsOfFile:rightPath];
    if (!leftRaw || !rightRaw) return;
    NSUInteger leftExpected = LEFT_YUV_WIDTH * LEFT_YUV_HEIGHT * 3 / 2;
    NSUInteger rightExpected = RIGHT_YUV_WIDTH * RIGHT_YUV_HEIGHT * 3 / 2;
    NSData *leftData = (leftRaw.length > leftExpected) ? [leftRaw subdataWithRange:NSMakeRange(0, leftExpected)] : leftRaw;
    NSData *rightData = (rightRaw.length > rightExpected) ? [rightRaw subdataWithRange:NSMakeRange(0, rightExpected)] : rightRaw;

    self.leftRGBA = [self rgba8TextureFromYUVData:leftData width:LEFT_YUV_WIDTH height:LEFT_YUV_HEIGHT];
    self.rightRGBA = [self rgba8TextureFromYUVData:rightData width:RIGHT_YUV_WIDTH height:RIGHT_YUV_HEIGHT];
    if (self.leftRGBA && self.rightRGBA) {
        NSData *leftRef = nil, *rightRef = nil;
        self.leftCGImage = [self cgImageFromRGBA8Texture:self.leftRGBA imageDataRef:&leftRef];
        self.rightCGImage = [self cgImageFromRGBA8Texture:self.rightRGBA imageDataRef:&rightRef];
        self.leftImageData = leftRef;
        self.rightImageData = rightRef;
        NSLog(@"✅ Stitch images ready");
    }
}

- (id<MTLTexture>)rgba8TextureFromYUVData:(NSData *)yuvData width:(NSUInteger)width height:(NSUInteger)height {
    const uint8_t *bytes = (const uint8_t *)yuvData.bytes;
    NSUInteger ySize = width * height;
    NSUInteger uvSize = (width/2) * (height/2);
    const uint8_t *yPlane = bytes, *vPlane = bytes + ySize, *uPlane = bytes + ySize + uvSize;
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> tex = [self.device newTextureWithDescriptor:desc];
    if (!tex) return nil;
    uint8_t *rgba = (uint8_t *)malloc(width * height * 4);
    for (NSUInteger y=0; y<height; ++y) {
        for (NSUInteger x=0; x<width; ++x) {
            NSUInteger idx = y*width + x;
            NSUInteger uvidx = (y/2)*(width/2) + (x/2);
            uint8_t Y = yPlane[idx], U = uPlane[uvidx], V = vPlane[uvidx];
            float r = Y + 1.402f*(V-128);
            float g = Y - 0.344f*(U-128) - 0.714f*(V-128);
            float b = Y + 1.772f*(U-128);
            rgba[idx*4+0] = (uint8_t)MAX(0, MIN(255, r));
            rgba[idx*4+1] = (uint8_t)MAX(0, MIN(255, g));
            rgba[idx*4+2] = (uint8_t)MAX(0, MIN(255, b));
            rgba[idx*4+3] = 255;
        }
    }
    [tex replaceRegion:MTLRegionMake2D(0,0,width,height) mipmapLevel:0 withBytes:rgba bytesPerRow:width*4];
    free(rgba);
    return tex;
}

- (CGImageRef)cgImageFromRGBA8Texture:(id<MTLTexture>)texture imageDataRef:(NSData **)outData {
    NSUInteger w = texture.width, h = texture.height;
    NSUInteger bpr = w * 4;
    NSMutableData *data = [NSMutableData dataWithLength:bpr * h];
    [texture getBytes:data.mutableBytes bytesPerRow:bpr fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];
    if (outData) *outData = data;
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef prov = CGDataProviderCreateWithCFData((CFDataRef)data);
    CGImageRef img = CGImageCreate(w, h, 8, 32, bpr, cs, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault, prov, NULL, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(prov);
    CGColorSpaceRelease(cs);
    return img;
}

- (void)performStitch {
    if (!self.leftRGBA || !self.rightRGBA) { NSLog(@"⚠️ Stitch images not loaded"); return; }
    HomographyResult *hResult = [HomographyEngine computeHomographyFromImageA:self.leftCGImage imageB:self.rightCGImage];
    if (!hResult.success) { [self fallbackStitch]; return; }
    NSLog(@"✅ H = [%f %f %f; %f %f %f; %f %f %f]",
          hResult.H.columns[0][0], hResult.H.columns[1][0], hResult.H.columns[2][0],
          hResult.H.columns[0][1], hResult.H.columns[1][1], hResult.H.columns[2][1],
          hResult.H.columns[0][2], hResult.H.columns[1][2], hResult.H.columns[2][2]);
    id<MTLCommandBuffer> cb = [self.commandQueue commandBuffer];
    self.stitchResult = [self.stitcher stitchTextureA:self.leftRGBA textureB:self.rightRGBA homography:hResult.H canvasWidth:hResult.canvasWidth canvasHeight:hResult.canvasHeight offsetX:hResult.offsetX offsetY:hResult.offsetY commandBuffer:cb];
    [cb commit];
    [cb waitUntilCompleted];
    if (self.stitchResult) {
        self.stitchReady = YES;
        NSLog(@"✅ Stitch completed, result size: %lu x %lu", (unsigned long)self.stitchResult.width, (unsigned long)self.stitchResult.height);
    } else {
        self.stitchReady = NO;
        NSLog(@"❌ Stitch failed");
    }
}

- (void)fallbackStitch {
    simd_float3x3 H = matrix_identity_float3x3;
    H.columns[2][0] = LEFT_YUV_WIDTH;
    NSUInteger cw = LEFT_YUV_WIDTH + RIGHT_YUV_WIDTH;
    NSUInteger ch = MAX(LEFT_YUV_HEIGHT, RIGHT_YUV_HEIGHT);
    id<MTLCommandBuffer> cb = [self.commandQueue commandBuffer];
    self.stitchResult = [self.stitcher stitchTextureA:self.leftRGBA textureB:self.rightRGBA homography:H canvasWidth:cw canvasHeight:ch offsetX:0 offsetY:0 commandBuffer:cb];
    [cb commit];
    [cb waitUntilCompleted];
    self.stitchReady = (self.stitchResult != nil);
}

#pragma mark - MTKViewDelegate
- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    [self.srEngine resetHistory];
    self.stitchReady = NO;
    self.stitchResult = nil;
}

- (void)drawInMTKView:(MTKView *)view {
    // ---------- 1. 局部超分模式 ----------
    if (self.localZoomMode && !CGRectIsEmpty(self.selectedRect)) {
        [self renderLocalZoomInView:view];
        return;
    }

    // ---------- 2. 拼接模式 ----------
    if (self.currentMode == DisplayModeStitch && self.stitchReady && self.stitchResult) {
        [self renderRGBA:self.stitchResult inView:view];
        return;
    }

    // ---------- 3. 全图超分/原始显示 ----------
    if (!self.yTexture) return;
    CGSize targetSize = (self.customScale > 0) ? CGSizeMake(YUV_WIDTH * self.customScale, YUV_HEIGHT * self.customScale) : view.drawableSize;
    id<MTLCommandBuffer> cb = [self.commandQueue commandBuffer];
    id<MTLTexture> outU = nil, outV = nil;
    id<MTLTexture> yOut = [self.srEngine processYTexture:self.yTexture uTexture:self.uTexture vTexture:self.vTexture targetSize:targetSize commandBuffer:cb outUTexture:&outU outVTexture:&outV];
    [cb commit];
    [cb waitUntilCompleted];
    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    if (!rp) return;
    id<MTLCommandBuffer> renderCB = [self.commandQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [renderCB renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:self.yuvPipelineState];
    [enc setVertexBuffer:self.vertexBuffer offset:0 atIndex:0];
    [enc setFragmentTexture:yOut atIndex:0];
    [enc setFragmentTexture:outU atIndex:1];
    [enc setFragmentTexture:outV atIndex:2];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    [renderCB presentDrawable:view.currentDrawable];
    [renderCB commit];
}

// ---------- 局部超分渲染 ----------
- (void)renderLocalZoomInView:(MTKView *)view {
    if (!self.yTexture) return;

    id<MTLCommandBuffer> cb = [self.commandQueue commandBuffer];

    // 1. 从当前帧裁剪出 ROI 对应的 Y/U/V 子纹理（GPU blit）
    id<MTLTexture> cropY, cropU, cropV;
    [self extractYTexture:self.yTexture
                 uTexture:self.uTexture
                 vTexture:self.vTexture
               fromRegion:self.selectedRect
                    outY:&cropY outU:&cropU outV:&cropV
           commandBuffer:cb];

    // 2. 子纹理送入超分引擎，放大到整个 drawable 尺寸
    CGSize targetSize = view.drawableSize;
    id<MTLTexture> outU = nil, outV = nil;
    id<MTLTexture> yOut = [self.srEngine processYTexture:cropY
                                                uTexture:cropU
                                                vTexture:cropV
                                              targetSize:targetSize
                                          commandBuffer:cb
                                            outUTexture:&outU
                                             outVTexture:&outV];

    [cb commit];
    [cb waitUntilCompleted]; // 确保超分结果可用（可优化为异步）

    // 3. 渲染超分后的 YUV 到屏幕
    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    if (!rp) return;

    id<MTLCommandBuffer> renderCB = [self.commandQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [renderCB renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:self.yuvPipelineState];
    [enc setVertexBuffer:self.vertexBuffer offset:0 atIndex:0];
    [enc setFragmentTexture:yOut atIndex:0];
    [enc setFragmentTexture:outU atIndex:1];
    [enc setFragmentTexture:outV atIndex:2];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    [renderCB presentDrawable:view.currentDrawable];
    [renderCB commit];
}

// GPU 纹理裁剪（YUV420 对齐）
- (void)extractYTexture:(id<MTLTexture>)srcY
               uTexture:(id<MTLTexture>)srcU
               vTexture:(id<MTLTexture>)srcV
             fromRegion:(CGRect)region
                  outY:(id<MTLTexture> *)outY
                  outU:(id<MTLTexture> *)outU
                  outV:(id<MTLTexture> *)outV
          commandBuffer:(id<MTLCommandBuffer>)cb {
    NSUInteger yW = (NSUInteger)CGRectGetWidth(region);
    NSUInteger yH = (NSUInteger)CGRectGetHeight(region);
    NSUInteger uvW = yW / 2;
    NSUInteger uvH = yH / 2;

    MTLTextureDescriptor *yDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                     width:yW height:yH mipmapped:NO];
    yDesc.usage = MTLTextureUsageShaderRead;
    *outY = [self.device newTextureWithDescriptor:yDesc];

    MTLTextureDescriptor *uvDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                      width:uvW height:uvH mipmapped:NO];
    uvDesc.usage = MTLTextureUsageShaderRead;
    *outU = [self.device newTextureWithDescriptor:uvDesc];
    *outV = [self.device newTextureWithDescriptor:uvDesc];

    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

    MTLOrigin yOrigin = MTLOriginMake(region.origin.x, region.origin.y, 0);
    MTLSize ySize = MTLSizeMake(yW, yH, 1);
    [blit copyFromTexture:srcY sourceSlice:0 sourceLevel:0 sourceOrigin:yOrigin sourceSize:ySize
                toTexture:*outY destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];

    MTLOrigin uvOrigin = MTLOriginMake(region.origin.x/2, region.origin.y/2, 0);
    MTLSize uvSize = MTLSizeMake(uvW, uvH, 1);
    [blit copyFromTexture:srcU sourceSlice:0 sourceLevel:0 sourceOrigin:uvOrigin sourceSize:uvSize
                toTexture:*outU destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [blit copyFromTexture:srcV sourceSlice:0 sourceLevel:0 sourceOrigin:uvOrigin sourceSize:uvSize
                toTexture:*outV destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];

    [blit endEncoding];
}

- (void)renderRGBA:(id<MTLTexture>)texture inView:(MTKView *)view {
    if (!texture || !self.rgbaPipelineState) return;
    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    if (!rp) return;
    id<MTLCommandBuffer> cb = [self.commandQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:self.rgbaPipelineState];
    [enc setVertexBuffer:self.vertexBuffer offset:0 atIndex:0];
    [enc setFragmentTexture:texture atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    [cb presentDrawable:view.currentDrawable];
    [cb commit];
}

- (NSUInteger)outputWidth  { return self.metalView.drawableSize.width; }
- (NSUInteger)outputHeight { return self.metalView.drawableSize.height; }

@end
