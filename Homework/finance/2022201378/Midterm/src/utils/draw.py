import jieba
import folium
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from folium.plugins import HeatMap
plt.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = ['STFangsong']

# 画出热力图并保存为 HTML 文件
def save_map(data,col_name=['lat','lon'],filename='heatmap.html',add_points=True):
    coords = data[col_name].dropna()
    center_lat = coords[col_name[0]].mean()
    center_lon = coords[col_name[1]].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')

    HeatMap(data=coords.values, radius=10, blur=15).add_to(m)

    if add_points:
        for idx, row in coords.iterrows():
            folium.CircleMarker(
                location=[row[col_name[0]], row[col_name[1]]],
                radius=3,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6
            ).add_to(m)
    m.save(filename)

def save_all_maps(train_price, test_price, train_rent, test_rent):
    save_map(train_price, col_name=['lat', 'lon'], filename='train_price_heatmap.html', add_points=True)
    save_map(test_price, col_name=['lat', 'lon'], filename='test_price_heatmap.html', add_points=True)
    save_map(train_rent, col_name=['lat', 'lon'], filename='train_rent_heatmap.html', add_points=True)
    save_map(test_rent, col_name=['lat', 'lon'], filename='test_rent_heatmap.html', add_points=True)


