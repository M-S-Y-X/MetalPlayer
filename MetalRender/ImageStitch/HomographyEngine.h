//
//  HomographyEngine.h
//  计算单应矩阵（Objective-C 接口）
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <simd/simd.h>

NS_ASSUME_NONNULL_BEGIN

@interface HomographyResult : NSObject
@property (nonatomic, assign) simd_float3x3 H;      // B → A 单应矩阵
@property (nonatomic, assign) NSUInteger canvasWidth;
@property (nonatomic, assign) NSUInteger canvasHeight;
@property (nonatomic, assign) int offsetX;
@property (nonatomic, assign) int offsetY;
@property (nonatomic, assign) BOOL success;
@end

@interface HomographyEngine : NSObject

+ (HomographyResult *)computeHomographyFromImageA:(CGImageRef)imageA
                                            imageB:(CGImageRef)imageB;

@end

NS_ASSUME_NONNULL_END
