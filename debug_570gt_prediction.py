#!/usr/bin/env python3
"""
调试570GT预测逻辑，分析为什么预测结果不合理
"""
import pandas as pd
from ml_final import FinalUltimateMcLarenPredictor

def debug_570gt_prediction():
    print("🔍 570GT 预测逻辑调试")
    print("=" * 50)
    
    # 加载实际数据
    df = pd.read_csv('mclaren_us_processed_final.csv')
    gt_data = df[df['model'] == '570GT']
    
    print(f"📊 570GT 实际市场数据:")
    print(f"   样本数量: {len(gt_data)}")
    print(f"   平均价格: ${gt_data['price'].mean():,.0f}")
    print(f"   价格范围: ${gt_data['price'].min():,.0f} - ${gt_data['price'].max():,.0f}")
    
    # 找到接近我们测试场景的实际样本
    print(f"\n🎯 寻找相似样本:")
    
    # 场景1：2017年，30k里程，密苏里州
    similar_1 = gt_data[
        (gt_data['year'] == 2017) & 
        (gt_data['mileage'] >= 25000) & 
        (gt_data['mileage'] <= 35000)
    ]
    print(f"   2017年30k左右里程: {len(similar_1)} 样本")
    if len(similar_1) > 0:
        print(f"   实际价格范围: ${similar_1['price'].min():,.0f} - ${similar_1['price'].max():,.0f}")
        print(f"   平均价格: ${similar_1['price'].mean():,.0f}")
    
    # 特别查看密苏里州的样本
    mo_samples = gt_data[gt_data['state'] == 'MO']
    print(f"   密苏里州样本: {len(mo_samples)} 个")
    if len(mo_samples) > 0:
        print(f"   密苏里州价格: {mo_samples[['year', 'mileage', 'price']].to_string()}")
    
    # 场景2：60k里程样本
    high_mileage = gt_data[gt_data['mileage'] >= 50000]
    print(f"   高里程(50k+)样本: {len(high_mileage)} 个")
    
    print(f"\n🤖 测试模型预测:")
    
    # 初始化预测器
    predictor = FinalUltimateMcLarenPredictor()
    
    # 只运行数据清理和特征工程，不训练模型
    predictor.df = pd.read_csv('mclaren_us_processed_final.csv')
    predictor.ultimate_data_cleaning()
    df_features = predictor.ultimate_feature_engineering()
    
    print(f"   清理后数据量: {len(predictor.df)}")
    
    # 检查570GT在清理后是否还存在
    gt_after_clean = predictor.df[predictor.df['model'] == '570GT']
    print(f"   清理后570GT样本: {len(gt_after_clean)}")
    
    if len(gt_after_clean) > 0:
        print(f"   清理后价格范围: ${gt_after_clean['price'].min():,.0f} - ${gt_after_clean['price'].max():,.0f}")
        
        # 检查特征工程后的数据
        gt_features = df_features[df_features['model'] == '570GT']
        print(f"   570GT模型分类: {gt_features['model_tier'].unique()}")
        print(f"   570GT里程类别: {gt_features['mileage_category'].unique()}")
        print(f"   570GT状况评分: {gt_features['condition_score'].unique()}")
    
    # 检查分段逻辑
    predictor.create_ultra_fine_segments(df_features)
    
    print(f"\n📊 分段信息:")
    for segment_name, segment_info in predictor.segments.items():
        if '570' in segment_name or 'sports' in segment_name:
            print(f"   {segment_name}: {segment_info['count']} 样本, 类型: {segment_info['type']}")
    
    print(f"\n⚠️  问题分析:")
    print(f"1. 570GT被归类为'sports'级别，可能与570S混在同一个分段")
    print(f"2. 样本量少(20个)，在数据清理中可能被大幅减少")
    print(f"3. 模型可能没有充分区分570GT与570S的差异")
    print(f"4. 30k里程在570GT中属于高里程，但模型可能没有正确处理")

if __name__ == "__main__":
    debug_570gt_prediction() 