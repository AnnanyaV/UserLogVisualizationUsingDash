from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
import sklearn
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

# import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud
import matplotlib.pyplot as plt



app = Dash(__name__)


df=pd.read_csv("check-1.csv")
df2=pd.read_csv("check-2.csv")
dfdocs= pd.read_json('Documents_Dataset_1.json')
df['ID']= [x.replace(' ', '') for x in df['ID']]
df['ID']= df['ID'].str.lower()
dfdocs['ID']= dfdocs['ID'].str.lower()
print(df['ID'])
dfcontents= pd.merge(df, dfdocs, on='ID')
# print(dfcontents.head())
datasets=[df, df2]
dff = df[df['segment'] == 0]
figscatter = go.Figure()
figscatter.add_trace(go.Scatter(x=dff['time'], y=dff['InteractionType'],
                    mode='markers',
                    name='markers'))

text=''
dfwc=dfcontents

# for i in range(1, len(dfwc)):
#   if 'Armsdealing' in dfwc.iloc[i, 3]:
#     dfwc.loc[i, 'contents']= dfwc.loc[i, 'Text'].replace(' ', '')
with open('readme.txt', 'w') as f:    
  for i in dfwc['title']:
    text=''.join(i)
    f.write(text)
    f.write(' ')
# with open('readme.txt', 'w') as f:    
#     for i in dfwc['Text']:
#       text=''.join(i)
#       f.write(text)
#       f.write(' ')

text = open("readme.txt", mode="r").read()
if '<br>' in text:
    text=text.replace('<br>', '')

text=[text]

# corpus = ['Hi what are you accepting here do you accept me',
# 'What are you thinking about getting today',
# 'Give me your password to get accepted into this school',
# 'The man went to the tree to get his sword back',
# 'go away to a far away place in a foreign land']

vectorizer = TfidfVectorizer(stop_words='english')
vecs = vectorizer.fit_transform(text)
feature_names = vectorizer.get_feature_names()
dense = vecs.todense()
lst1 = dense.tolist()
df = pd.DataFrame(lst1, columns=feature_names)
df.T.sum(axis=1)
Cloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(df.T.sum(axis=1))

# stopwords = STOPWORDS
# wc = WordCloud(background_color="white", stopwords=stopwords, height=600, width=400)
# wc.generate(text)
# wc.to_file("assets/wordcloud_output.png")

plt.imshow(Cloud)
plt.show()



app.layout = html.Div(children=[
    html.H1(children='User logs visualization'),
    html.Img(src=app.get_asset_url('wordcloud_output.png')),


    dcc.Graph(
            id='crossfilter-indicator-scatter',
            figure=figscatter,
            
        ),

    html.Div(dcc.Slider(
        df['segment'].min(),
        df['segment'].max(),
        step=None,
        id='crossfilter-year--slider',
        value=df['segment'].max(),
        marks={str(year): str(year) for year in df['segment'].unique()}
    ),
    ),
    

], )

@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    Input('crossfilter-year--slider', 'value'),)

def updateScatter(value):
    dff = df[df['segment'] == value]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff['time'], y=dff['InteractionType'],
                    mode='markers',
                    name='markers'))

    return fig




if __name__ == '__main__':
    app.run_server(debug=True)



