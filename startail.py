#!/usr/bin/env python3
"""
星轨动画生成器 - 纯净版
只生成干净的星轨动画和纯星轨图片，无任何文字信息
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import datetime

def create_clean_star_trail():
    """创建纯净的星轨动画"""
    
    print("=" * 60)
    print("✨ 星轨动画生成器 - 纯净版 ✨")
    print("=" * 60)
    
    # 获取用户输入
    if len(sys.argv) < 3:
        print("使用方法:")
        print("  python star_trail_clean.py <照片文件夹> <输出视频>")
        print("")
        print("示例:")
        print("  python star_trail_clean.py ./星空照片 star_trail.mp4")
        print("")
        print("可选参数:")
        print("  --fps 24        # 视频帧率 (默认: 25)")
        print("  --max 200       # 最大处理图片数 (默认: 全部)")
        print("  --hold 2        # 最后画面保持秒数 (默认: 2)")
        print("  --bright 0.8    # 亮度系数 (0.5-1.5, 默认: 0.8)")
        return
    
    # 解析参数
    input_folder = sys.argv[1]
    output_video = sys.argv[2]
    fps = 25
    max_images = None
    hold_seconds = 2  # 缩短保持时间
    brightness_factor = 0.8
    
    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--fps" and i+1 < len(sys.argv):
            fps = int(sys.argv[i+1])
        elif sys.argv[i] == "--max" and i+1 < len(sys.argv):
            max_images = int(sys.argv[i+1])
        elif sys.argv[i] == "--hold" and i+1 < len(sys.argv):
            hold_seconds = int(sys.argv[i+1])
        elif sys.argv[i] == "--bright" and i+1 < len(sys.argv):
            brightness_factor = float(sys.argv[i+1])
    
    # 检查输入文件夹
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ 错误: 文件夹 '{input_folder}' 不存在！")
        return
    
    # 查找图片文件
    print(f"📁 正在搜索文件夹: {input_folder}")
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.tif', '.tiff']:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print("❌ 错误: 未找到任何图片文件！")
        return
    
    # 按文件名排序
    image_files = sorted(image_files)
    
    # 限制图片数量
    if max_images and len(image_files) > max_images:
        print(f"📊 将处理前 {max_images} 张图片（共 {len(image_files)} 张）")
        image_files = image_files[:max_images]
    else:
        print(f"📊 找到 {len(image_files)} 张图片")
    
    # 读取第一张图片获取尺寸
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print(f"❌ 错误: 无法读取图片 {image_files[0]}")
        return
    
    original_height, original_width = first_img.shape[:2]
    print(f"📐 图片原始尺寸: {original_width}x{original_height}")
    
    # 确定输出尺寸（保持比例）
    max_width = 1920
    if original_width > max_width:
        scale = max_width / original_width
        width = max_width
        height = int(original_height * scale)
    else:
        width = original_width
        height = original_height
    
    # 确保尺寸是偶数
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1
    
    print(f"🎬 输出视频尺寸: {width}x{height}")
    print(f"⏱️  视频帧率: {fps} fps")
    print(f"💡 亮度系数: {brightness_factor}")
    
    # 创建输出目录
    video_path = Path(output_video)
    output_dir = video_path.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # 生成纯星轨图片文件名
    if output_video.endswith('.mp4'):
        image_filename = output_video.replace('.mp4', '_star_trail.jpg')
    elif output_video.endswith('.avi'):
        image_filename = output_video.replace('.avi', '_star_trail.jpg')
    else:
        image_filename = f"{output_video}_star_trail.jpg"
    
    print(f"🖼️  星轨图片将保存为: {image_filename}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video.isOpened():
        print("❌ 错误: 无法创建视频文件！")
        return
    
    print("\n🚀 开始生成纯净星轨动画...")
    print("💡 视频中将不包含任何文字信息")
    
    # 初始化星轨累积
    star_trail = np.zeros((height, width, 3), dtype=np.float32)
    
    # 处理每张图片 - 纯净版，无任何文字
    for i, img_file in enumerate(tqdm(image_files, desc="🔄 处理图片")):
        # 读取图片
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        # 调整尺寸
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # 纯变亮混合（永久保持轨迹）
        star_trail = np.maximum(star_trail, img.astype(np.float32))
        
        # 创建当前帧 - 纯净，无文字
        current_frame = np.clip(star_trail * brightness_factor, 0, 255).astype(np.uint8)
        
        # 写入视频帧
        video.write(current_frame)
    
    # 获取最终星轨画面 - 纯净版
    final_trail = np.clip(star_trail * brightness_factor, 0, 255).astype(np.uint8)
    
    # 保存纯星轨图片（无任何文字）
    print("\n💾 保存纯星轨图片...")
    cv2.imwrite(image_filename, final_trail, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 保持最终画面几秒 - 纯净版，无文字无特效
    print("\n⏳ 生成视频结尾...")
    hold_frames = fps * hold_seconds
    for _ in tqdm(range(hold_frames), desc="生成结尾帧"):
        video.write(final_trail)  # 直接写入最终画面，无任何文字
    
    # 释放资源
    video.release()
    
    print("\n" + "=" * 60)
    print("✅ 纯净版处理完成！")
    print("=" * 60)
    
    # 输出总结
    print(f"\n🎬 视频文件:")
    print(f"  📁 {output_video}")
    if os.path.exists(output_video):
        video_size = os.path.getsize(output_video) / (1024 * 1024)
        print(f"  💾 大小: {video_size:.1f} MB")
        print(f"  ⏱️  时长: {len(image_files)/fps + hold_seconds:.1f} 秒")
    
    print(f"\n🖼️  星轨图片:")
    print(f"  📁 {image_filename}")
    if os.path.exists(image_filename):
        img_size = os.path.getsize(image_filename) / 1024
        print(f"  💾 大小: {img_size:.1f} KB")
    
    print(f"\n📊 处理统计:")
    print(f"  🖼️  处理图片: {len(image_files)} 张")
    print(f"  🖥️  输出尺寸: {width}x{height}")
    print(f"  🌟 轨迹效果: 永久保持，纯净无文字")
    
    print("\n✨ 特点:")
    print("  • 视频无任何文字信息")
    print("  • 图片为纯星轨，无任何水印")
    print("  • 星轨永久累积，不会消失")
    print("=" * 60)

def show_help():
    """显示帮助信息"""
    print("""
星轨动画生成器 - 纯净版
    
特点:
• 视频纯净无任何文字信息
• 图片为纯星轨，无水印无文字
• 星轨永久保持，不会消失
• 输出文件简洁

使用方法:
    python star_trail_clean.py <照片文件夹> <输出视频> [选项]

示例:
    python star_trail_clean.py ./星空照片 ./星轨.mp4
    python star_trail_clean.py ./photos ./star_trail.mp4 --fps 30

选项:
    --fps 24        视频帧率 (默认: 25)
    --max 200       最大处理图片数 (默认: 全部)
    --hold 2        最后画面保持秒数 (默认: 2)
    --bright 0.8    亮度系数 0.5-1.5 (默认: 0.8)

亮度系数说明:
    0.5 - 较暗，适合亮星多的照片
    0.8 - 适中，适合大多数情况
    1.2 - 较亮，适合暗星多的照片

输出文件:
    星轨.mp4              # 纯净星轨动画视频
    星轨_star_trail.jpg    # 纯星轨图片，无任何文字

安装依赖:
    pip install opencv-python numpy tqdm
    """)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_help()
    elif len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h", "/?"]:
        show_help()
    else:
        create_clean_star_trail()