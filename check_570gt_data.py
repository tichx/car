import pandas as pd

# 加载数据
df = pd.read_csv('mclaren_us_processed_final.csv')

print('=== 570GT 数据分析 ===')
gt_data = df[df['model'] == '570GT']
print(f'570GT 样本数量: {len(gt_data)}')

if len(gt_data) > 0:
    print(f'年份范围: {gt_data["year"].min()} - {gt_data["year"].max()}')
    print(f'价格范围: ${gt_data["price"].min():,.0f} - ${gt_data["price"].max():,.0f}')
    print(f'里程数范围: {gt_data["mileage"].min():,.0f} - {gt_data["mileage"].max():,.0f}')
    print(f'平均价格: ${gt_data["price"].mean():,.0f}')
    print(f'价格标准差: ${gt_data["price"].std():,.0f}')
    print('\n=== 570GT 样本详情 ===')
    print(gt_data[['year', 'mileage', 'price', 'state']].to_string())
else:
    print('没有找到570GT数据')

print('\n=== 相关模型数据对比 ===')
related_models = ['570S', '570GT', '570S Spider', 'GT']
for model in related_models:
    model_data = df[df['model'] == model]
    if len(model_data) > 0:
        print(f'{model}: {len(model_data)} 样本, 平均价格: ${model_data["price"].mean():,.0f}')

print('\n=== 运动级模型 (sports tier) 分析 ===')
sports_models = ['570S', '570GT', '570S Spider', '650S', '650S Spider']
for model in sports_models:
    model_data = df[df['model'] == model]
    if len(model_data) > 0:
        print(f'{model}: {len(model_data)} 样本')
        # 查看2017年的数据
        model_2017 = model_data[model_data['year'] == 2017]
        if len(model_2017) > 0:
            print(f'  2017年款: {len(model_2017)} 样本, 价格范围: ${model_2017["price"].min():,.0f} - ${model_2017["price"].max():,.0f}')
            # 查看高里程数样本
            high_mileage = model_2017[model_2017['mileage'] >= 25000]
            if len(high_mileage) > 0:
                print(f'  高里程(25k+): {len(high_mileage)} 样本, 平均价格: ${high_mileage["price"].mean():,.0f}') 