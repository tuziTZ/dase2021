import pandas as pd
import plotly.graph_objects as go
#Clock frequency is approx. 2500.0 MHz
#Memory mountain (MB/sec)
# 数据预处理
df = pd.read_fwf('out.txt')
df.to_csv('data.csv', index=False)

data = pd.read_csv('data.csv')
size = data['size']
data = data.drop(columns='size', axis=1)
#绘图
fig = go.Figure(data=[go.Surface(z=data.values, x=data.columns, y=size)])

fig.update_layout(title='My memory mountain', autosize=False,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.write_html('mm.html') # 保存HTML
fig.show()
#512k：高速缓存大小
