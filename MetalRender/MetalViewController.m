//
//  MetalViewController.m
//  MetalRender
//

#import "MetalViewController.h"
#import "SREngine.h"

#define YUV_WIDTH       1280
#define YUV_HEIGHT      1280
#define PLAYBACK_FPS    30
#define YUV_FILE_PATH   [[NSBundle mainBundle] pathForResource:@"1280_1280" ofType:@"yuv"]

typedef struct {
    vector_float2 position;
    vector_float2 texCoord;
} Vertex;

@interface MetalViewController () <MTKViewDelegate>
@property (nonatomic, strong) MTKView *metalView;
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLRenderPipelineState> pipelineState;
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
@end

@implementation MetalViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.device = MTLCreateSystemDefaultDevice();
    if (!self.device) { NSLog(@"Metal not supported"); return; }
    
    self.metalView = [[MTKView alloc] initWithFrame:self.view.bounds device:self.device];
    self.metalView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.metalView.delegate = self;
    self.metalView.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    [self.view addSubview:self.metalView];
    
    self.commandQueue = [self.device newCommandQueue];
    self.customScale = 0.0;
    
    self.srEngine = [[SREngine alloc] initWithDevice:self.device];
    
    [self setupRenderPipeline];
    [self setupVertexBuffer];
    [self loadYUVFileAndPrepareTextures];
    [self setupUI];
}

- (void)dealloc { [self.playbackTimer invalidate]; }

#pragma mark - UI

- (void)setupUI {
    CGFloat yPos = self.view.bounds.size.height - 50;
    
    self.modeButton = [[NSButton alloc] initWithFrame:NSMakeRect(20, yPos, 200, 30)];
    self.modeButton.title = @"Mode: IBP+Temporal";
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
}

- (void)cycleMode:(NSButton *)sender {
    NSInteger totalModes = 5;
    NSInteger newMode = (self.srEngine.mode + 1) % totalModes;
    self.srEngine.mode = (SRMode)newMode;
    switch (self.srEngine.mode) {
        case SRModeOff:            sender.title = @"Mode: Off (Bilinear)"; break;
        case SRModeLanczos:        sender.title = @"Mode: Lanczos only"; break;
        case SRModeIBP:            sender.title = @"Mode: IBP (single)"; break;
        case SRModeTemporalIBP:    sender.title = @"Mode: IBP+Temporal"; break;
        case SRModeDeepLearning:   sender.title = @"Mode: DeepLearning"; break;
    }
    [self.srEngine resetHistory];
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

#pragma mark - Metal setup

- (void)setupRenderPipeline {
    id<MTLLibrary> lib = [self.device newDefaultLibrary];
    id<MTLFunction> vert = [lib newFunctionWithName:@"vertex_main"];
    id<MTLFunction> frag = [lib newFunctionWithName:@"fragment_yuv_to_rgb"];
    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vert;
    desc.fragmentFunction = frag;
    desc.colorAttachments[0].pixelFormat = self.metalView.colorPixelFormat;
    NSError *err;
    self.pipelineState = [self.device newRenderPipelineStateWithDescriptor:desc error:&err];
    if (!self.pipelineState) NSLog(@"Render pipeline error: %@", err);
}

- (void)setupVertexBuffer {
    static const Vertex vertices[] = {
        {{-1.0,  1.0}, {0.0, 0.0}}, {{ 1.0,  1.0}, {1.0, 0.0}},
        {{-1.0, -1.0}, {0.0, 1.0}}, {{ 1.0, -1.0}, {1.0, 1.0}},
        {{ 1.0,  1.0}, {1.0, 0.0}}, {{-1.0, -1.0}, {0.0, 1.0}}
    };
    self.vertexBuffer = [self.device newBufferWithBytes:vertices length:sizeof(vertices)
                                                options:MTLResourceStorageModeShared];
}

#pragma mark - YUV 纹理加载

- (void)createTextures {
    MTLTextureDescriptor *yDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                     width:YUV_WIDTH height:YUV_HEIGHT mipmapped:NO];
    yDesc.usage = MTLTextureUsageShaderRead;
    self.yTexture = [self.device newTextureWithDescriptor:yDesc];
    
    MTLTextureDescriptor *uvDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                      width:YUV_WIDTH/2 height:YUV_HEIGHT/2 mipmapped:NO];
    uvDesc.usage = MTLTextureUsageShaderRead;
    self.uTexture = [self.device newTextureWithDescriptor:uvDesc];
    self.vTexture = [self.device newTextureWithDescriptor:uvDesc];
}

- (void)loadYUVFileAndPrepareTextures {
    NSString *path = YUV_FILE_PATH;
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        NSLog(@"YUV file not found"); return;
    }
    NSError *err;
    self.fullYUVData = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:&err];
    if (err || !self.fullYUVData) { NSLog(@"Read error: %@", err); return; }
    
    NSInteger ySize = YUV_WIDTH * YUV_HEIGHT;
    NSInteger uvSize = (YUV_WIDTH/2) * (YUV_HEIGHT/2);
    self.frameSize = ySize + 2 * uvSize;
    self.totalFrames = self.fullYUVData.length / self.frameSize;
    if (self.totalFrames == 0) return;
    
    [self createTextures];
    self.currentFrame = 0;
    [self updateTexturesForFrame:self.currentFrame];
    
    if (self.totalFrames > 1) {
        self.playbackTimer = [NSTimer scheduledTimerWithTimeInterval:1.0/PLAYBACK_FPS
                                                              target:self selector:@selector(nextFrame)
                                                            userInfo:nil repeats:YES];
    }
    [self.metalView setNeedsDisplay:YES];
}

- (void)updateTexturesForFrame:(NSInteger)frameIndex {
    if (!self.fullYUVData || frameIndex >= self.totalFrames) return;
    const uint8_t *bytes = (const uint8_t *)self.fullYUVData.bytes + frameIndex * self.frameSize;
    NSInteger ySize = YUV_WIDTH * YUV_HEIGHT;
    NSInteger uvSize = (YUV_WIDTH/2) * (YUV_HEIGHT/2);
    
    [self.yTexture replaceRegion:MTLRegionMake2D(0, 0, YUV_WIDTH, YUV_HEIGHT)
                     mipmapLevel:0 withBytes:bytes bytesPerRow:YUV_WIDTH];
    [self.uTexture replaceRegion:MTLRegionMake2D(0, 0, YUV_WIDTH/2, YUV_HEIGHT/2)
                     mipmapLevel:0 withBytes:bytes + ySize + uvSize bytesPerRow:YUV_WIDTH/2];
    [self.vTexture replaceRegion:MTLRegionMake2D(0, 0, YUV_WIDTH/2, YUV_HEIGHT/2)
                     mipmapLevel:0 withBytes:bytes + ySize bytesPerRow:YUV_WIDTH/2];
}

- (void)nextFrame {
    self.currentFrame = (self.currentFrame + 1) % self.totalFrames;
    [self updateTexturesForFrame:self.currentFrame];
    dispatch_async(dispatch_get_main_queue(), ^{ [self.metalView setNeedsDisplay:YES]; });
}

#pragma mark - MTKViewDelegate

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    [self.srEngine resetHistory];
}

- (void)drawInMTKView:(MTKView *)view {
    if (!self.yTexture) return;
    CGSize drawableSize = view.drawableSize;
    if (drawableSize.width <= 0 || drawableSize.height <= 0) return;
    
    CGSize targetSize;
    if (self.customScale > 0) {
        targetSize = CGSizeMake(YUV_WIDTH * self.customScale, YUV_HEIGHT * self.customScale);
    } else {
        targetSize = drawableSize;
    }
    
    id<MTLCommandBuffer> cb = [self.commandQueue commandBuffer];
    id<MTLTexture> outU = nil, outV = nil;
    id<MTLTexture> yOut = [self.srEngine processYTexture:self.yTexture
                                                 uTexture:self.uTexture
                                                 vTexture:self.vTexture
                                               targetSize:targetSize
                                            commandBuffer:cb
                                             outUTexture:&outU
                                             outVTexture:&outV];
    [cb commit];
    [cb waitUntilCompleted];
    
    id<MTLCommandBuffer> renderCB = [self.commandQueue commandBuffer];
    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    if (!rp) return;
    id<MTLRenderCommandEncoder> enc = [renderCB renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:self.pipelineState];
    [enc setVertexBuffer:self.vertexBuffer offset:0 atIndex:0];
    [enc setFragmentTexture:yOut atIndex:0];
    [enc setFragmentTexture:outU atIndex:1];
    [enc setFragmentTexture:outV atIndex:2];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    [renderCB presentDrawable:view.currentDrawable];
    [renderCB commit];
}

@end
