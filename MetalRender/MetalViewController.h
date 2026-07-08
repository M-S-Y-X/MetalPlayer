//
//  MetalViewController.h
//  Metal YUV Player
//
//  Created on 2026-07-08.
//

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Metal YUV 播放器视图控制器
 * 支持 YV12 格式视频文件的循环播放
 */
@interface MetalViewController : NSViewController <MTKViewDelegate>

@end

NS_ASSUME_NONNULL_END
