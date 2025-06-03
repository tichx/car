#!/usr/bin/env python3
"""
测试McLaren 570GT的两个具体场景：
1. 2017年McLaren 570GT，30,000英里，密苏里州，当前价值
2. 2027年（3年后）McLaren 570GT，60,000英里，加州，预期价值
"""

import sys
import os
sys.path.append(os.getcwd())

from ml_final import FinalUltimateMcLarenPredictor

def test_570gt_scenarios():
    """测试两个McLaren 570GT场景"""
    print("🏎️ McLaren 570GT 价值评估测试")
    print("=" * 60)
    
    # 初始化预测器并训练模型
    print("🔧 初始化预测系统...")
    predictor = FinalUltimateMcLarenPredictor()
    evaluator = predictor.run_final_analysis()
    
    print("\n" + "=" * 60)
    print("🎯 570GT 具体场景测试")
    print("=" * 60)
    
    # 场景1：2017年McLaren 570GT，30,000英里，密苏里州，当前价值
    print("\n📍 场景1：当前价值评估")
    print("-" * 30)
    scenario1 = {
        'year': 2017,
        'model': '570GT', 
        'mileage': 30000,
        'state': 'MO',  # 密苏里州
        'transmission': 'Automatic'
    }
    
    result1 = evaluator.predict_price(**scenario1)
    
    print(f"🚗 车辆信息：{scenario1['year']} McLaren {scenario1['model']}")
    print(f"📏 里程数：{scenario1['mileage']:,} 英里")
    print(f"📍 地点：密苏里州 (MO)")
    print(f"📅 评估时间：2024年（当前）")
    print(f"💰 预估价值：${result1['price_prediction']:,.0f}")
    print(f"🎯 匹配段落：{result1['segment_used']}")
    print(f"🤖 使用模型：{result1['model_used']}")
    print(f"📊 置信度：{result1['confidence_score']}/4")
    print(f"📈 段落MAE：${result1['segment_mae']:,.0f}")
    
    # 场景2：2027年（3年后）McLaren 570GT，60,000英里，加州
    print("\n📍 场景2：3年后价值预测")
    print("-" * 30)
    scenario2 = {
        'year': 2017,  # 车辆制造年份仍然是2017
        'model': '570GT',
        'mileage': 60000,  # 3年后增加到60,000英里
        'state': 'CA',     # 加州
        'transmission': 'Automatic'
    }
    
    result2 = evaluator.predict_price(**scenario2)
    
    print(f"🚗 车辆信息：{scenario2['year']} McLaren {scenario2['model']}")
    print(f"📏 里程数：{scenario2['mileage']:,} 英里")
    print(f"📍 地点：加州 (CA)")
    print(f"📅 评估时间：2027年（3年后）")
    print(f"💰 预估价值：${result2['price_prediction']:,.0f}")
    print(f"🎯 匹配段落：{result2['segment_used']}")
    print(f"🤖 使用模型：{result2['model_used']}")
    print(f"📊 置信度：{result2['confidence_score']}/4")
    print(f"📈 段落MAE：${result2['segment_mae']:,.0f}")
    
    # 价值变化分析
    print("\n📊 价值变化分析")
    print("=" * 30)
    price_change = result2['price_prediction'] - result1['price_prediction']
    percentage_change = (price_change / result1['price_prediction']) * 100
    
    print(f"💰 当前价值 (MO, 30k miles)：${result1['price_prediction']:,.0f}")
    print(f"💰 3年后价值 (CA, 60k miles)：${result2['price_prediction']:,.0f}")
    print(f"📈 绝对变化：${price_change:+,.0f}")
    print(f"📊 百分比变化：{percentage_change:+.1f}%")
    
    if price_change > 0:
        print("✅ 预计价值上升")
    else:
        print("📉 预计价值下降")
    
    # 影响因素分析
    print("\n🔍 影响因素分析")
    print("=" * 30)
    print("1. 地理位置影响：")
    print("   • 密苏里州 → 加州：通常有正面影响（加州豪车市场更活跃）")
    print("2. 里程数影响：")
    print("   • 30,000 → 60,000英里：负面影响（里程数翻倍）")
    print("3. 时间影响：")
    print("   • 2024 → 2027：车龄增加3年，通常有负面影响")
    print("4. McLaren 570GT特点：")
    print("   • GT车型：相对保值，但受里程数影响较大")
    print("   • 2017年款：已过最快折旧期，折旧率相对稳定")

if __name__ == "__main__":
    test_570gt_scenarios() 