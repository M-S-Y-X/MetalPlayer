#import "AppDelegate.h"
#import "MetalViewController.h"

@interface AppDelegate ()
@property (strong, nonatomic) NSWindow *window;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // 创建窗口
    NSRect frame = NSMakeRect(0, 0, 1280, 1280);   // 窗口大小与 YUV 图像一致
    NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                                   styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
                                                     backing:NSBackingStoreBuffered
                                                       defer:NO];
    window.title = @"YUV Player (YV12)";
    [window makeKeyAndOrderFront:nil];
    
    // 设置 Metal 视图控制器
    MetalViewController *vc = [[MetalViewController alloc] init];
    window.contentViewController = vc;
    
    self.window = window;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

@end
