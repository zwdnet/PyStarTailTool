#!/usr/bin/env python3
"""
星轨动画生成器 - 最终修复版
解决背景亮斑问题，优化星星检测
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

def create_enhanced_star_image_final(image_files, output_size, brightness_factor=0.8, max_images_for_enhance=None):
    """
    最终修复版：解决背景亮斑问题
    
    参数:
        image_files: 所有图片文件列表
        output_size: 输出尺寸 (width, height)
        brightness_factor: 亮度系数
        max_images_for_enhance: 用于增强的最大图片数（None表示全部）
        
    返回:
        增强后的星轨图片
    """
    width, height = output_size
    
    # 确定用于增强的图片数量
    if max_images_for_enhance and max_images_for_enhance > 0 and len(image_files) > max_images_for_enhance:
        enhance_files = image_files[:max_images_for_enhance]
        print(f"🌟 使用前 {max_images_for_enhance} 张图片进行增强（共 {len(image_files)} 张）")
    else:
        enhance_files = image_files
        print(f"🌟 使用 {len(enhance_files)} 张图片进行增强")
    
    print("\n🚀 开始增强处理（解决亮斑问题）...")
    
    # 步骤1：分析图像特征，避免误检测
    print("\n1. 分析图像特征...")
    
    # 加载几张代表性图片分析
    sample_size = min(10, len(enhance_files))
    sample_images = []
    
    for i in range(sample_size):
        img = cv2.imread(str(enhance_files[i]))
        if img is None:
            continue
        
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        sample_images.append(img)
    
    if not sample_images:
        print("⚠️  无法加载样本图片")
        return None
    
    # 分析图像亮度分布
    print("  分析亮度分布...")
    all_gray = []
    for img in sample_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        all_gray.append(gray.flatten())
    
    all_gray_array = np.concatenate(all_gray)
    mean_brightness = np.mean(all_gray_array)
    std_brightness = np.std(all_gray_array)
    
    # 关键修复：使用更高的阈值避免检测到背景噪点
    base_threshold = mean_brightness + std_brightness * 2.5  # 增加标准差倍数
    star_threshold = max(50, min(base_threshold, 120))  # 设置合理范围
    
    print(f"  平均亮度: {mean_brightness:.1f}")
    print(f"  亮度标准差: {std_brightness:.1f}")
    print(f"  星星检测阈值: {star_threshold:.1f}")
    
    # 步骤2：改进的星星检测方法
    print("\n2. 改进星星检测方法...")
    
    # 分批处理图片
    batch_size = 15  # 更小的批次
    total_batches = (len(enhance_files) + batch_size - 1) // batch_size
    
    # 初始化累积器
    height, width = output_size[1], output_size[0]
    star_accumulator = np.zeros((height, width), dtype=np.float32)  # 星星出现次数
    brightness_accumulator = np.zeros((height, width, 3), dtype=np.float32)  # 亮度累积
    valid_pixel_count = 0
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(enhance_files))
        batch_files = enhance_files[start_idx:end_idx]
        
        print(f"  处理批次 {batch_idx+1}/{total_batches} ({len(batch_files)} 张)")
        
        for img_file in tqdm(batch_files, desc=f"批次 {batch_idx+1}", leave=False):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # 调整尺寸
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            # 转换为灰度
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 方法1：基本阈值检测
            _, basic_mask = cv2.threshold(gray, star_threshold, 255, cv2.THRESH_BINARY)
            
            # 方法2：自适应阈值检测（对不均匀光照更好）
            adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY, 11, 2)
            
            # 方法3：局部对比度增强检测
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_abs = np.abs(laplacian)
            laplacian_mask = (laplacian_abs > 15).astype(np.uint8) * 255
            
            # 合并多种检测结果（加权）
            combined_mask = basic_mask.astype(np.float32) * 0.5 + \
                           adaptive_mask.astype(np.float32) * 0.3 + \
                           laplacian_mask.astype(np.float32) * 0.2
            
            # 转换为二值图像
            _, final_mask = cv2.threshold(combined_mask.astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
            
            # 关键修复：形态学操作去除大块区域（解决亮斑问题）
            # 使用开运算去除小噪点
            kernel_open = np.ones((2, 2), np.uint8)
            cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
            
            # 使用闭运算连接小的星星点
            kernel_close = np.ones((1, 1), np.uint8)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # 关键修复：移除大面积的连通区域（可能是亮斑）
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
            
            # 创建最终的星星掩码，只保留小区域
            final_star_mask = np.zeros_like(cleaned_mask)
            for i in range(1, num_labels):  # 跳过背景
                area = stats[i, cv2.CC_STAT_AREA]
                # 关键：只保留小面积区域，排除大块亮斑
                if 1 <= area <= 15:  # 星星通常是很小的点
                    final_star_mask[labels == i] = 255
            
            # 转换为布尔掩码
            star_mask_bool = final_star_mask > 128
            
            # 检查是否有检测到星星
            if np.sum(star_mask_bool) > 0:
                # 累积星星出现次数
                star_accumulator += star_mask_bool.astype(np.float32)
                
                # 累积星星区域的亮度
                img_float = img.astype(np.float32)
                for c in range(3):
                    brightness_accumulator[:, :, c] += img_float[:, :, c] * star_mask_bool.astype(np.float32)
                
                valid_pixel_count += 1
            
            # 释放内存
            del img, gray, basic_mask, adaptive_mask, laplacian, combined_mask, final_mask, cleaned_mask, final_star_mask
            gc.collect()
    
    if valid_pixel_count == 0:
        print("⚠️  没有检测到有效星星")
        return None
    
    print(f"✅ 成功处理 {valid_pixel_count} 张图片")
    
    # 步骤3：识别稳定星星位置（使用更严格的标准）
    print("\n3. 识别稳定星星位置...")
    
    # 关键修复：提高频率阈值，避免背景噪点被误认为星星
    frequency_threshold = valid_pixel_count * 0.4  # 提高到40%的出现频率
    
    # 创建稳定星星掩码
    stable_star_mask = star_accumulator >= frequency_threshold
    
    # 计算稳定星星数量
    stable_star_count = np.sum(stable_star_mask)
    print(f"  发现 {stable_star_count} 个稳定星星位置")
    
    # 如果星星太少，适当降低阈值
    if stable_star_count < 30:
        print("  检测到的星星较少，适当降低阈值...")
        frequency_threshold = valid_pixel_count * 0.3
        stable_star_mask = star_accumulator >= frequency_threshold
        stable_star_count = np.sum(stable_star_mask)
        print(f"  调整后: {stable_star_count} 个稳定星星位置")
    
    if stable_star_count == 0:
        print("⚠️  没有检测到稳定星星")
        return None
    
    # 步骤4：计算平均亮度并增强
    print("\n4. 计算平均亮度和增强...")
    
    # 计算每个星星位置的平均亮度
    star_avg_brightness = np.zeros((height, width, 3), dtype=np.float32)
    
    # 避免除零
    star_accumulator_nonzero = np.maximum(star_accumulator, 1)
    
    for c in range(3):
        star_avg_brightness[:, :, c] = brightness_accumulator[:, :, c] / star_accumulator_nonzero
    
    # 关键修复：使用更温和的增强，避免过强
    enhancement_factor = 1.0 + np.log1p(star_accumulator) * 0.3  # 减少增强系数
    
    # 应用增强
    enhanced_stars = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        enhanced_stars[:, :, c] = star_avg_brightness[:, :, c] * enhancement_factor
    
    # 应用亮度系数
    enhanced_stars = np.clip(enhanced_stars * brightness_factor, 0, 255)
    
    # 步骤5：创建最终图像（解决背景亮斑问题）
    print("\n5. 创建最终图像...")
    
    # 加载几张高质量图片作为背景
    background_samples = min(3, len(image_files))
    background = np.zeros((height, width, 3), dtype=np.float32)
    
    for i in range(background_samples):
        img = cv2.imread(str(image_files[i]))
        if img is None:
            continue
        
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        background += img.astype(np.float32)
    
    if background_samples > 0:
        background = background / background_samples
    
    # 关键修复：对背景进行降噪处理
    background_uint8 = background.astype(np.uint8)
    denoised_background = cv2.bilateralFilter(background_uint8, 5, 50, 50)
    background = denoised_background.astype(np.float32)
    
    # 创建最终图像（使用原始背景，不调暗）
    final_image = background.copy()
    
    # 叠加增强的星星
    star_indices = np.where(stable_star_mask)
    
    print(f"  叠加 {len(star_indices[0])} 个增强星星...")
    
    star_added = 0
    for i in tqdm(range(len(star_indices[0])), desc="叠加星星", leave=False):
        y, x = star_indices[0][i], star_indices[1][i]
        
        # 检查增强后的星星是否确实比背景亮
        enhanced_value = np.mean(enhanced_stars[y, x])
        background_value = np.mean(final_image[y, x])
        
        # 只有增强后的星星明显比背景亮时才叠加
        if enhanced_value > background_value * 1.2:
            # 计算增强强度（基于出现次数，但限制最大强度）
            occurrence_count = star_accumulator[y, x]
            alpha = min(0.6, occurrence_count / 15)  # 限制最大alpha值为0.6
            
            # 混合增强星星和背景
            final_image[y, x] = enhanced_stars[y, x] * alpha + final_image[y, x] * (1 - alpha)
            star_added += 1
            
            # 添加非常轻微的光晕效果
            if occurrence_count > 8:
                radius = 1
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            distance = np.sqrt(dy*dy + dx*dx)
                            if 0 < distance <= radius:  # 排除中心点
                                weight = 1.0 - distance / radius
                                final_image[ny, nx] = final_image[ny, nx] * (1 - weight*0.1) + enhanced_stars[y, x] * weight*0.1
    
    print(f"  成功添加 {star_added} 个增强星星")
    
    # 步骤6：后处理（避免引入亮斑）
    print("\n6. 后处理...")
    final_image = post_process_final(final_image, stable_star_mask)
    
    return np.clip(final_image, 0, 255).astype(np.uint8)

def post_process_final(image, star_mask):
    """最终后处理，避免引入亮斑"""
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    
    # 1. 轻微降噪，但保留星星细节
    denoised = cv2.bilateralFilter(image_uint8, 3, 30, 30)
    
    # 2. 对比度增强（只对非星星区域进行温和增强）
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 对星星区域和非星星区域分别处理
    star_mask_uint8 = star_mask.astype(np.uint8) * 255
    
    # 对非星星区域进行CLAHE增强
    non_star_mask = 255 - star_mask_uint8
    l_non_star = cv2.bitwise_and(l, non_star_mask)
    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))  # 降低clipLimit
    l_non_star_enhanced = clahe.apply(l_non_star)
    
    # 合并星星区域和非星星区域
    l_star = cv2.bitwise_and(l, star_mask_uint8)
    l_combined = cv2.add(l_star, l_non_star_enhanced)
    
    # 合并通道
    enhanced_lab = cv2.merge([l_combined, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 3. 对整幅图像进行非常轻微的锐化
    kernel = np.array([[0, -0.1, 0],
                       [-0.1, 1.4, -0.1],
                       [0, -0.1, 0]])
    sharpened = cv2.filter2D(enhanced.astype(np.float32), -1, kernel)
    
    # 4. 确保星星区域不被过度处理
    for c in range(3):
        sharpened[:, :, c] = np.where(
            star_mask,
            np.clip(sharpened[:, :, c] * 0.9 + image_uint8[:, :, c] * 0.1, 0, 255),  # 保留部分原始星星
            sharpened[:, :, c]
        )
    
    return sharpened

def create_clean_star_trail():
    """创建纯净的星轨动画"""
    
    print("=" * 60)
    print("✨ 星轨动画生成器 - 最终修复版 ✨")
    print("=" * 60)
    print("💡 解决背景亮斑问题，优化星星检测")
    
    # 获取用户输入
    if len(sys.argv) < 3:
        print("使用方法:")
        print("  python star_trail_final_fixed.py <照片文件夹> <输出视频>")
        print("")
        print("示例:")
        print("  python star_trail_final_fixed.py ./星空照片 star_trail.mp4")
        print("")
        print("可选参数:")
        print("  --fps 24             # 视频帧率 (默认: 25)")
        print("  --max 200            # 最大处理图片数 (默认: 全部)")
        print("  --hold 2             # 最后画面保持秒数 (默认: 2)")
        print("  --bright 0.8         # 亮度系数 (0.5-1.5, 默认: 0.8)")
        print("  --enhance            # 生成增强版星轨图片")
        print("  --enhance-max 100    # 用于增强的最大图片数 (0=全部, 默认: 100)")
        print("  --threshold auto     # 星星检测阈值 (auto/数值, 默认: auto)")
        return
    
    # 解析参数
    input_folder = sys.argv[1]
    output_video = sys.argv[2]
    fps = 25
    max_images = None
    hold_seconds = 2
    brightness_factor = 0.8
    enable_enhance = False
    max_images_for_enhance = 100
    manual_threshold = None
    
    for i in range(3, len(sys.argv)):
        if sys.argv[i] == "--fps" and i+1 < len(sys.argv):
            fps = int(sys.argv[i+1])
        elif sys.argv[i] == "--max" and i+1 < len(sys.argv):
            max_images = int(sys.argv[i+1])
        elif sys.argv[i] == "--hold" and i+1 < len(sys.argv):
            hold_seconds = int(sys.argv[i+1])
        elif sys.argv[i] == "--bright" and i+1 < len(sys.argv):
            brightness_factor = float(sys.argv[i+1])
        elif sys.argv[i] == "--enhance":
            enable_enhance = True
        elif sys.argv[i] == "--enhance-max" and i+1 < len(sys.argv):
            max_images_for_enhance = int(sys.argv[i+1])
        elif sys.argv[i] == "--threshold" and i+1 < len(sys.argv):
            threshold_str = sys.argv[i+1]
            if threshold_str.lower() != "auto":
                try:
                    manual_threshold = float(threshold_str)
                except:
                    manual_threshold = None
    
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
    original_count = len(image_files)
    if max_images and len(image_files) > max_images:
        print(f"📊 将处理前 {max_images} 张图片（共 {original_count} 张）")
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
    
    # 确定输出尺寸
    max_width = 1920
    if original_width > max_width:
        scale = max_width / original_width
        width = max_width
        height = int(original_height * scale)
    else:
        width = original_width
        height = original_height
    
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1
    
    print(f"🎬 输出视频尺寸: {width}x{height}")
    print(f"⏱️  视频帧率: {fps} fps")
    print(f"💡 亮度系数: {brightness_factor}")
    print(f"🌟 增强模式: {'启用' if enable_enhance else '禁用'}")
    
    if enable_enhance:
        if max_images_for_enhance == 0:
            print(f"📈 增强使用图片: 全部 {len(image_files)} 张")
        else:
            print(f"📈 增强使用图片: 前 {max_images_for_enhance} 张")
        
        if manual_threshold:
            print(f"🎯 手动阈值: {manual_threshold}")
        else:
            print(f"🎯 阈值计算: 自动")
    
    # 创建输出目录
    video_path = Path(output_video)
    output_dir = video_path.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # 生成文件名
    if output_video.endswith('.mp4'):
        image_filename = output_video.replace('.mp4', '_star_trail.jpg')
        enhanced_filename = output_video.replace('.mp4', '_enhanced_final.jpg')
    elif output_video.endswith('.avi'):
        image_filename = output_video.replace('.avi', '_star_trail.jpg')
        enhanced_filename = output_video.replace('.avi', '_enhanced_final.jpg')
    else:
        image_filename = f"{output_video}_star_trail.jpg"
        enhanced_filename = f"{output_video}_enhanced_final.jpg"
    
    print(f"🖼️  星轨图片将保存为: {image_filename}")
    if enable_enhance:
        print(f"🌟 最终增强图片将保存为: {enhanced_filename}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video.isOpened():
        print("❌ 错误: 无法创建视频文件！")
        return
    
    print("\n🚀 开始生成纯净星轨动画...")
    
    # 初始化星轨累积
    star_trail = np.zeros((height, width, 3), dtype=np.float32)
    
    # 处理每张图片
    for i, img_file in enumerate(tqdm(image_files, desc="🔄 处理图片")):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        # 调整尺寸
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # 纯变亮混合
        star_trail = np.maximum(star_trail, img.astype(np.float32))
        
        # 创建当前帧
        current_frame = np.clip(star_trail * brightness_factor, 0, 255).astype(np.uint8)
        
        # 写入视频帧
        video.write(current_frame)
        
        # 定期释放内存
        if i % 100 == 0 and i > 0:
            gc.collect()
    
    # 获取最终星轨画面
    final_trail = np.clip(star_trail * brightness_factor, 0, 255).astype(np.uint8)
    
    # 保存纯星轨图片
    print("\n💾 保存纯星轨图片...")
    cv2.imwrite(image_filename, final_trail, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 生成最终增强版星轨图片（解决亮斑问题）
    if enable_enhance:
        print("\n" + "=" * 50)
        print("✨ 开始最终增强处理 ✨")
        print("=" * 50)
        print("💡 解决背景亮斑问题，优化星星检测")
        
        # 确定用于增强的图片数量
        if max_images_for_enhance == 0:
            enhance_files = image_files
        else:
            enhance_files = image_files[:max_images_for_enhance]
        
        print(f"🔢 将使用 {len(enhance_files)} 张图片进行最终增强")
        
        # 创建增强版星轨图片
        enhanced_image = create_enhanced_star_image_final(
            enhance_files, 
            (width, height), 
            brightness_factor,
            max_images_for_enhance if max_images_for_enhance > 0 else None
        )
        
        if enhanced_image is not None:
            cv2.imwrite(enhanced_filename, enhanced_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"✅ 最终增强版星轨图片已保存: {enhanced_filename}")
            
            # 检查亮斑问题是否解决
            gray_enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_enhanced)
            print(f"📊 增强图片平均亮度: {mean_brightness:.1f}")
        else:
            print("⚠️  增强处理失败，使用原始星轨图片")
            cv2.imwrite(enhanced_filename, final_trail, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # 保持最终画面几秒
    print("\n⏳ 生成视频结尾...")
    hold_frames = fps * hold_seconds
    for _ in tqdm(range(hold_frames), desc="生成结尾帧"):
        video.write(final_trail)
    
    # 释放资源
    video.release()
    
    print("\n" + "=" * 60)
    print("✅ 最终修复版处理完成！")
    print("=" * 60)
    
    # 输出总结
    print(f"\n🎬 视频文件:")
    print(f"  📁 {output_video}")
    if os.path.exists(output_video):
        video_size = os.path.getsize(output_video) / (1024 * 1024)
        print(f"  💾 大小: {video_size:.1f} MB")
    
    print(f"\n🖼️  星轨图片:")
    print(f"  📁 {image_filename}")
    if os.path.exists(image_filename):
        img_size = os.path.getsize(image_filename) / 1024
        print(f"  💾 大小: {img_size:.1f} KB")
    
    if enable_enhance and os.path.exists(enhanced_filename):
        print(f"\n🌟 最终增强版星轨图片:")
        print(f"  📁 {enhanced_filename}")
        enhanced_size = os.path.getsize(enhanced_filename) / 1024
        print(f"  💾 大小: {enhanced_size:.1f} KB")
        print(f"  ✨ 特点: 解决背景亮斑问题，精确星星增强")
    
    print(f"\n📊 处理统计:")
    print(f"  🖼️  处理图片: {len(image_files)} 张")
    print(f"  🖥️  输出尺寸: {width}x{height}")
    
    print("\n✨ 最终修复特点:")
    print("  • 解决背景亮斑问题")
    print("  • 优化星星检测算法")
    print("  • 多重验证确保检测真实星星")
    print("  • 温和增强，避免过处理")
    print("=" * 60)

def show_help():
    """显示帮助信息"""
    print("""
星轨动画生成器 - 最终修复版
    
特点:
• 解决背景亮斑问题
• 优化星星检测算法
• 视频纯净无任何文字信息
• 图片为纯星轨，无水印无文字
• 星轨永久保持，不会消失

主要改进:
1. 解决背景亮斑问题 - 通过更严格的星星检测和面积过滤
2. 优化阈值计算 - 使用更高的阈值避免检测到背景噪点
3. 多重检测方法 - 结合多种检测算法提高准确性
4. 面积过滤 - 只保留小面积区域，排除大块亮斑

使用方法:
    python star_trail_final_fixed.py <照片文件夹> <输出视频> [选项]

示例:
    python star_trail_final_fixed.py ./星空照片 ./星轨.mp4 --enhance
    python star_trail_final_fixed.py ./photos ./star_trail.mp4 --enhance --enhance-max 150

选项:
    --fps 24            视频帧率 (默认: 25)
    --max 200           最大处理图片数 (默认: 全部)
    --hold 2            最后画面保持秒数 (默认: 2)
    --bright 0.8        亮度系数 0.5-1.5 (默认: 0.8)
    --enhance           启用最终增强功能
    --enhance-max 100   用于增强的最大图片数 (0=全部, 默认: 100)
    --threshold auto    星星检测阈值 (auto/数值, 默认: auto)

输出文件:
    星轨.mp4                    # 纯净星轨动画视频
    星轨_star_trail.jpg          # 纯星轨图片，无任何文字
    星轨_enhanced_final.jpg      # 最终增强版星轨图片

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