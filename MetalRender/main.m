#import <Cocoa/Cocoa.h>   // 引入 Cocoa 框架，用于 macOS 界面开发
#import "AppDelegate.h"   // 引入自定义的应用委托类头文件

int main(int argc, const char * argv[]) {
    @autoreleasepool {  // 自动释放池，管理内存
        // 获取应用程序的单例实例
        NSApplication *app = [NSApplication sharedApplication];
        // 创建 AppDelegate 实例，作为应用的委托对象
        AppDelegate *delegate = [[AppDelegate alloc] init];
        // 将委托设置给应用，应用会回调委托中的方法（如 applicationDidFinishLaunching:）
        app.delegate = delegate;
        // 启动应用的主事件循环，开始处理用户交互和系统事件
        [app run];
    }
    return 0;
}
