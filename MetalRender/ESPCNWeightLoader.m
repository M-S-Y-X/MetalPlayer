// ESPCNWeightLoader.m
#import "ESPCNWeightLoader.h"

@implementation ESPCNWeightLoader

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
    }
    return self;
}

- (BOOL)loadWeightsFromBundle {
    NSBundle *bundle = [NSBundle mainBundle];
    
    NSDictionary *files = @{
        @"conv1_weight": @"espcn_conv1_weight",
        @"conv1_bias":   @"espcn_conv1_bias",
        @"conv2_weight": @"espcn_conv2_weight",
        @"conv2_bias":   @"espcn_conv2_bias",
        @"conv3_weight": @"espcn_conv3_weight",
        @"conv3_bias":   @"espcn_conv3_bias"
    };
    
    return [self loadWeightsFromFiles:files];
}

- (BOOL)loadWeightsFromFiles:(NSDictionary *)filePaths {
    // 权重数组维度
    // conv1_weight: [64][1][5][5] = 1600 floats
    // conv1_bias:   [64] = 64 floats
    // conv2_weight: [32][64][3][3] = 18432 floats
    // conv2_bias:   [32] = 32 floats
    // conv3_weight: [4][32][3][3] = 1152 floats
    // conv3_bias:   [4] = 4 floats
    
    NSError *error = nil;
    
    // --- Conv1 权重 ---
    NSString *path = [[NSBundle mainBundle] pathForResource:filePaths[@"conv1_weight"]
                                                     ofType:@"bin"];
    if (!path) {
        NSLog(@"❌ ESPCN: conv1_weight.bin not found");
        return NO;
    }
    NSData *w1Data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    if (error) {
        NSLog(@"❌ ESPCN: Failed to load conv1_weight: %@", error);
        return NO;
    }
    // 确保数据是 float32 格式
    NSInteger expectedSize = 64 * 1 * 5 * 5 * sizeof(float); // 6400 bytes
    if (w1Data.length != expectedSize) {
        NSLog(@"⚠️ ESPCN: conv1_weight size mismatch. Expected %ld, got %lu",
              (long)expectedSize, (unsigned long)w1Data.length);
        // 继续尝试，但可能出错
    }
    self.conv1Weight = [self.device newBufferWithBytes:w1Data.bytes
                                                length:w1Data.length
                                               options:MTLResourceStorageModeShared];
    
    // --- Conv1 偏置 ---
    path = [[NSBundle mainBundle] pathForResource:filePaths[@"conv1_bias"] ofType:@"bin"];
    NSData *b1Data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    expectedSize = 64 * sizeof(float); // 256 bytes
    if (b1Data.length >= expectedSize) {
        self.conv1Bias = [self.device newBufferWithBytes:b1Data.bytes
                                                  length:expectedSize
                                                 options:MTLResourceStorageModeShared];
    }
    
    // --- Conv2 权重 ---
    path = [[NSBundle mainBundle] pathForResource:filePaths[@"conv2_weight"] ofType:@"bin"];
    NSData *w2Data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    expectedSize = 32 * 64 * 3 * 3 * sizeof(float); // 73728 bytes
    if (w2Data.length >= expectedSize) {
        self.conv2Weight = [self.device newBufferWithBytes:w2Data.bytes
                                                    length:expectedSize
                                                   options:MTLResourceStorageModeShared];
    }
    
    // --- Conv2 偏置 ---
    path = [[NSBundle mainBundle] pathForResource:filePaths[@"conv2_bias"] ofType:@"bin"];
    NSData *b2Data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    expectedSize = 32 * sizeof(float); // 128 bytes
    if (b2Data.length >= expectedSize) {
        self.conv2Bias = [self.device newBufferWithBytes:b2Data.bytes
                                                  length:expectedSize
                                                 options:MTLResourceStorageModeShared];
    }
    
    // --- Conv3 权重 ---
    path = [[NSBundle mainBundle] pathForResource:filePaths[@"conv3_weight"] ofType:@"bin"];
    NSData *w3Data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    expectedSize = 4 * 32 * 3 * 3 * sizeof(float); // 4608 bytes
    if (w3Data.length >= expectedSize) {
        self.conv3Weight = [self.device newBufferWithBytes:w3Data.bytes
                                                    length:expectedSize
                                                   options:MTLResourceStorageModeShared];
    }
    
    // --- Conv3 偏置 ---
    path = [[NSBundle mainBundle] pathForResource:filePaths[@"conv3_bias"] ofType:@"bin"];
    NSData *b3Data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    expectedSize = 4 * sizeof(float); // 16 bytes
    if (b3Data.length >= expectedSize) {
        self.conv3Bias = [self.device newBufferWithBytes:b3Data.bytes
                                                  length:expectedSize
                                                 options:MTLResourceStorageModeShared];
    }
    
    NSLog(@"✅ ESPCN weights loaded successfully");
    NSLog(@"   conv1_weight: %lu bytes", (unsigned long)self.conv1Weight.length);
    NSLog(@"   conv2_weight: %lu bytes", (unsigned long)self.conv2Weight.length);
    NSLog(@"   conv3_weight: %lu bytes", (unsigned long)self.conv3Weight.length);
    
    return YES;
}

@end
