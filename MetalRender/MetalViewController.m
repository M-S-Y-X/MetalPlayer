//
//  MetalViewController.m
//  MetalRender
//

#import "MetalViewController.h"

// ---------- 用户配置 ----------
#define YUV_WIDTH       1280
#define YUV_HEIGHT      1280
#define PLAYBACK_FPS    30
#define YUV_FILE_PATH   [[NSBundle mainBundle] pathForResource:@"1280_1280" ofType:@"yuv"]
// -----------------------------

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
@end

@implementation MetalViewController

#pragma mark - 生命周期

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.device = MTLCreateSystemDefaultDevice();
    if (!self.device) {
        NSLog(@"Metal is not supported on this device");
        return;
    }
    
    self.metalView = [[MTKView alloc] initWithFrame:self.view.bounds device:self.device];
    self.metalView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.metalView.delegate = self;
    self.metalView.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    [self.view addSubview:self.metalView];
    
    self.commandQueue = [self.device newCommandQueue];
    
    [self setupRenderPipeline];
    [self setupVertexBuffer];
    [self loadYUVFileAndPrepareTextures];
}

- (void)dealloc {
    [self.playbackTimer invalidate];
}

#pragma mark - Metal 初始化

- (void)setupRenderPipeline {
    NSError *error = nil;
    id<MTLLibrary> library = [self.device newDefaultLibrary];
    if (!library) {
        NSLog(@"Failed to load default library");
        return;
    }
    
    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"fragment_yuv_to_rgb"];
    
    if (!vertexFunction || !fragmentFunction) {
        NSLog(@"Failed to load shader functions");
        return;
    }
    
    MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunction;
    pipelineDesc.fragmentFunction = fragmentFunction;
    pipelineDesc.colorAttachments[0].pixelFormat = self.metalView.colorPixelFormat;
    
    self.pipelineState = [self.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!self.pipelineState) {
        NSLog(@"Failed to create pipeline state: %@", error);
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
    self.vertexBuffer = [self.device newBufferWithBytes:vertices
                                                 length:sizeof(vertices)
                                                options:MTLResourceStorageModeShared];
}

#pragma mark - YUV 文件加载与多帧播放

- (void)loadYUVFileAndPrepareTextures {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *filePath = YUV_FILE_PATH;
    BOOL isDir = NO;
    BOOL exists = [fm fileExistsAtPath:filePath isDirectory:&isDir];
    
    NSLog(@"===== YUV File Load Debug =====");
    NSLog(@"File path: %@", filePath);
    NSLog(@"Exists: %d, isDirectory: %d", exists, isDir);
    
    if (!exists || isDir) {
        NSLog(@"❌ ERROR: File does not exist or is a directory.");
        return;
    }
    
    NSError *attrError = nil;
    NSDictionary *attrs = [fm attributesOfItemAtPath:filePath error:&attrError];
    if (attrError) {
        NSLog(@"❌ ERROR: Failed to get file attributes: %@", attrError);
        return;
    }
    
    unsigned long long fileSizeOnDisk = [attrs fileSize];
    NSLog(@"File size on disk: %llu bytes (%.2f MB)", fileSizeOnDisk, fileSizeOnDisk / (1024.0 * 1024.0));
    
    NSInteger yPlaneSize = YUV_WIDTH * YUV_HEIGHT;
    NSInteger uvPlaneSize = (YUV_WIDTH / 2) * (YUV_HEIGHT / 2);
    self.frameSize = yPlaneSize + 2 * uvPlaneSize;
    
    NSLog(@"Calculated frame size: %ld bytes", (long)self.frameSize);
    
    if (fileSizeOnDisk < self.frameSize) {
        NSLog(@"❌ ERROR: File size is smaller than one frame.");
        return;
    }
    
    NSError *readError = nil;
    self.fullYUVData = [NSData dataWithContentsOfFile:filePath
                                              options:NSDataReadingMappedIfSafe
                                                error:&readError];
    if (readError || !self.fullYUVData) {
        NSLog(@"❌ ERROR: Failed to read file: %@", readError);
        return;
    }
    
    NSUInteger readDataLength = self.fullYUVData.length;
    NSLog(@"Successfully read data length: %lu bytes", (unsigned long)readDataLength);
    
    self.totalFrames = readDataLength / self.frameSize;
    NSLog(@"Total frames: %ld", (long)self.totalFrames);
    
    [self createTextures];
    
    self.currentFrame = 0;
    [self updateTexturesForFrame:self.currentFrame];
    NSLog(@"✅ Initialized with first frame.");
    
    if (self.totalFrames > 1) {
        if (self.playbackTimer) {
            [self.playbackTimer invalidate];
        }
        self.playbackTimer = [NSTimer scheduledTimerWithTimeInterval:1.0/PLAYBACK_FPS
                                                              target:self
                                                            selector:@selector(nextFrame)
                                                            userInfo:nil
                                                             repeats:YES];
        NSLog(@"✅ Playback timer started.");
    }
    
    [self.metalView setNeedsDisplay:YES];
}

- (void)createTextures {
    // 使用 R8Unorm 格式，着色器可以直接读取为 float
    MTLTextureDescriptor *yDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                      width:YUV_WIDTH
                                                                                     height:YUV_HEIGHT
                                                                                  mipmapped:NO];
    yDesc.usage = MTLTextureUsageShaderRead;
    self.yTexture = [self.device newTextureWithDescriptor:yDesc];
    
    MTLTextureDescriptor *uDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                      width:YUV_WIDTH/2
                                                                                     height:YUV_HEIGHT/2
                                                                                  mipmapped:NO];
    uDesc.usage = MTLTextureUsageShaderRead;
    self.uTexture = [self.device newTextureWithDescriptor:uDesc];
    
    MTLTextureDescriptor *vDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                      width:YUV_WIDTH/2
                                                                                     height:YUV_HEIGHT/2
                                                                                  mipmapped:NO];
    vDesc.usage = MTLTextureUsageShaderRead;
    self.vTexture = [self.device newTextureWithDescriptor:vDesc];
}

- (void)updateTexturesForFrame:(NSInteger)frameIndex {
    if (!self.fullYUVData || frameIndex >= self.totalFrames)
        return;
    
    NSInteger offset = frameIndex * self.frameSize;
    const uint8_t *bytes = (const uint8_t *)self.fullYUVData.bytes + offset;
    
    NSInteger yPlaneSize = YUV_WIDTH * YUV_HEIGHT;
    NSInteger uvPlaneSize = (YUV_WIDTH / 2) * (YUV_HEIGHT / 2);
    
    // YV12 格式: Y 平面 -> V 平面 -> U 平面
    const uint8_t *yPlane = bytes;
    const uint8_t *vPlane = bytes + yPlaneSize;
    const uint8_t *uPlane = bytes + yPlaneSize + uvPlaneSize;
    
    MTLRegion yRegion = MTLRegionMake2D(0, 0, YUV_WIDTH, YUV_HEIGHT);
    [self.yTexture replaceRegion:yRegion
                     mipmapLevel:0
                       withBytes:yPlane
                     bytesPerRow:YUV_WIDTH];
    
    MTLRegion uvRegion = MTLRegionMake2D(0, 0, YUV_WIDTH/2, YUV_HEIGHT/2);
    [self.uTexture replaceRegion:uvRegion
                     mipmapLevel:0
                       withBytes:uPlane
                     bytesPerRow:YUV_WIDTH/2];
    
    [self.vTexture replaceRegion:uvRegion
                     mipmapLevel:0
                       withBytes:vPlane
                     bytesPerRow:YUV_WIDTH/2];
    
    if (frameIndex == 0) {
        NSLog(@"First Y pixel: %d, First V pixel: %d, First U pixel: %d",
              yPlane[0], vPlane[0], uPlane[0]);
    }
}

- (void)nextFrame {
    self.currentFrame++;
    if (self.currentFrame >= self.totalFrames) {
        self.currentFrame = 0;
    }
    [self updateTexturesForFrame:self.currentFrame];
    
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.metalView setNeedsDisplay:YES];
    });
}

#pragma mark - MTKViewDelegate

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    // 无需处理
}

- (void)drawInMTKView:(MTKView *)view {
    if (!self.yTexture || !self.uTexture || !self.vTexture) {
        return;
    }
    
    id<MTLCommandBuffer> commandBuffer = [self.commandQueue commandBuffer];
    MTLRenderPassDescriptor *renderPassDesc = view.currentRenderPassDescriptor;
    if (!renderPassDesc)
        return;
    
    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    [encoder setRenderPipelineState:self.pipelineState];
    [encoder setVertexBuffer:self.vertexBuffer offset:0 atIndex:0];
    
    [encoder setFragmentTexture:self.yTexture atIndex:0];
    [encoder setFragmentTexture:self.uTexture atIndex:1];
    [encoder setFragmentTexture:self.vTexture atIndex:2];
    
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    
    [encoder endEncoding];
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
}

@end
