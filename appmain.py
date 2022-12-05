from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import STOPWORDS, WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.io as pio
from collections import OrderedDict

app = Dash(__name__)

data = pd.read_json('Arms_P1_InteractionsLogs.json')
data2 = pd.read_json('Arms_P2_InteractionsLogs.json')
data3=pd.read_json('Arms_P3_InteractionsLogs.json')
data4=pd.read_json('Arms_P4_InteractionsLogs.json')
data5=pd.read_json('Arms_P5_InteractionsLogs.json')
data6=pd.read_json('Arms_P6_InteractionsLogs.json')
data7=pd.read_json('Arms_P7_InteractionsLogs.json')
data8=pd.read_json('Arms_P8_InteractionsLogs.json')

doc= pd.read_csv('/Users/annanya/Desktop/ArmsDealingVisualizationDemo/FirstDemoIV/doc.csv')
#PERSON 1
dataset = data
x_column = data['time']
y_column = data['time']
bubble_column = data['duration']
time_column = data['time']

data = data[data.InteractionType != 'Think_aloud']
data_max_scaled = data.copy()
# data_max_scaled['duration'] = data_max_scaled['duration']  / data_max_scaled['duration'].abs().max()
data_max_scaled['time']=data_max_scaled['time']/600
for i in range(1,len(data_max_scaled)):
    data_max_scaled.iloc[i, 0]=data_max_scaled.iloc[i, 4]-data_max_scaled.iloc[i-1, 4]

# data_max_scaled['time']=data_max_scaled['time']//60



#PERSON 2
dataset = data2
x_column = data2['time']
y_column = data2['time']
bubble_column = data2['duration']
time_column = data2['time']

data2 = data2[data2.InteractionType != 'Think_aloud']
data_max_scaled2 = data2.copy()
# data_max_scaled2['duration'] = data_max_scaled2['duration']  / data_max_scaled2['duration'].abs().max()
data_max_scaled2['time']=data_max_scaled2['time']/600
for i in range(1,len(data_max_scaled2)):
    data_max_scaled2.iloc[i, 0]=data_max_scaled2.iloc[i, 4]-data_max_scaled2.iloc[i-1, 4]

#PERSON 3
dataset = data3
x_column = data3['time']
y_column = data3['time']
bubble_column = data3['duration']
time_column = data3['time']

data3 = data3[data3.InteractionType != 'Think_aloud']
data_max_scaled3 = data3.copy()
# data_max_scaled3['duration'] = data_max_scaled3['duration']  / data_max_scaled3['duration'].abs().max()
data_max_scaled3['time']=data_max_scaled3['time']/600
for i in range(1,len(data_max_scaled3)):
    data_max_scaled3.iloc[i, 0]=data_max_scaled3.iloc[i, 4]-data_max_scaled3.iloc[i-1, 4]

#PERSON 4
dataset = data4
x_column = data4['time']
y_column = data4['time']
bubble_column = data4['duration']
time_column = data4['time']

data4 = data4[data4.InteractionType != 'Think_aloud']
data_max_scaled4 = data4.copy()
# data_max_scaled4['duration'] = data_max_scaled4['duration']  / data_max_scaled4['duration'].abs().max()
data_max_scaled4['time']=data_max_scaled4['time']/600
for i in range(1,len(data_max_scaled4)):
    data_max_scaled4.iloc[i, 0]=data_max_scaled4.iloc[i, 4]-data_max_scaled4.iloc[i-1, 4]

#PERSON 5
dataset = data5
x_column = data5['time']
y_column = data5['time']
bubble_column = data5['duration']
time_column = data5['time']

data5 = data5[data5.InteractionType != 'Think_aloud']
data_max_scaled5 = data5.copy()
# data_max_scaled5['duration'] = data_max_scaled5['duration']  / data_max_scaled5['duration'].abs().max()
data_max_scaled5['time']=data_max_scaled5['time']/600
for i in range(1,len(data_max_scaled5)):
    data_max_scaled5.iloc[i, 0]=data_max_scaled5.iloc[i, 4]-data_max_scaled5.iloc[i-1, 4]

#PERSON 6
dataset = data6
x_column = data6['time']
y_column = data6['time']
bubble_column = data6['duration']
time_column = data6['time']

data6 = data6[data6.InteractionType != 'Think_aloud']
data_max_scaled6 = data6.copy()
# data_max_scaled6['duration'] = data_max_scaled6['duration']  / data_max_scaled6['duration'].abs().max()
data_max_scaled6['time']=data_max_scaled6['time']/600
for i in range(1,len(data_max_scaled6)):
    data_max_scaled6.iloc[i, 0]=data_max_scaled6.iloc[i, 4]-data_max_scaled6.iloc[i-1, 4]

#PERSON 7
dataset = data7
x_column = data7['time']
y_column = data7['time']
bubble_column = data7['duration']
time_column = data7['time']

data7 = data7[data7.InteractionType != 'Think_aloud']
data_max_scaled7 = data7.copy()
# data_max_scaled7['duration'] = data_max_scaled7['duration']  / data_max_scaled7['duration'].abs().max()
data_max_scaled7['time']=data_max_scaled7['time']/600
for i in range(1,len(data_max_scaled7)):
    data_max_scaled7.iloc[i, 0]=data_max_scaled7.iloc[i, 4]-data_max_scaled7.iloc[i-1, 4]

#PERSON 8
dataset = data8
x_column = data8['time']
y_column = data8['time']
bubble_column = data8['duration']
time_column = data8['time']

data8 = data8[data8.InteractionType != 'Think_aloud']
data_max_scaled8 = data8.copy()
# data_max_scaled8['duration'] = data_max_scaled8['duration']  / data_max_scaled8['duration'].abs().max()
data_max_scaled8['time']=data_max_scaled8['time']/600
for i in range(1,len(data_max_scaled8)):
    data_max_scaled8.iloc[i, 0]=data_max_scaled8.iloc[i, 4]-data_max_scaled8.iloc[i-1, 4]

pio.templates.default = "plotly_white"


fig = go.Figure(data=go.Heatmap(
                   z=data_max_scaled['duration'],
                   x=data_max_scaled['time'],
                   y=data_max_scaled['InteractionType'],
                   colorscale="blues",
                   
                   hoverongaps = False,
                   hovertemplate =
                    "<b>%{x} </b><br><br>" +
                    "Interaction: %{y}<br>"
                    ))




d={}
for i in range(0,len(data_max_scaled)):
    if data_max_scaled.iloc[i, 3] in d:
        d[data_max_scaled.iloc[i, 3]]+= data_max_scaled.iloc[i, 0]
    else:
        d[data_max_scaled.iloc[i, 3]]= data_max_scaled.iloc[i, 0]



figrow = go.Figure(go.Bar(
            x=list(d.values()),
            y=list(d.keys()),
            orientation='h'))


#Word cloud

dfdocs= pd.read_json('Documents_Dataset_1.json')

data_max_scaled['ID']= [str(x).replace(' ', '') for x in data_max_scaled['ID']]
data_max_scaled['ID']= data_max_scaled['ID'].str.lower()
dfdocs['ID']= dfdocs['ID'].str.lower()
dfcontents= pd.merge(data_max_scaled, dfdocs, on='ID')


text=''
dfwc=dfcontents

with open('readme.txt', 'w') as f:    
  for i in dfwc['title']:
    text=''.join(i)
    f.write(text)
    f.write(' ')


text = open("readme.txt", mode="r").read()
if '<br>' in text:
    text=text.replace('<br>', '')

text=[text]

vectorizer = TfidfVectorizer(stop_words='english')
vecs = vectorizer.fit_transform(text)
feature_names = vectorizer.get_feature_names()
dense = vecs.todense()
lst1 = dense.tolist()
df = pd.DataFrame(lst1, columns=feature_names)
df.T.sum(axis=1)
Cloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(df.T.sum(axis=1))


d={}
for i in range(0,len(data_max_scaled)):
    if data_max_scaled.iloc[i, 3] in d:
        d[data_max_scaled.iloc[i, 3]]+= data_max_scaled.iloc[i, 0]
    else:
        d[data_max_scaled.iloc[i, 3]]= data_max_scaled.iloc[i, 0]
if '0' in d.keys():
    d.pop('0', 'None')
d.pop('bottom-up', None)
d.pop('top-down', None)

d1 = sorted(d.items(), key=lambda x:x[1], reverse=True)
d = dict(d1)
data={'documents':list(d.keys()), 'duration': list(d.values())}
new = pd.DataFrame.from_dict(data)
fig = px.line_polar(new.loc[0:10, :], r='duration', theta='documents', line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r)


app.layout = html.Div(children=[
    
    html.Div(children=[
        html.H2(children='User log visualizations'),
        html.H3(children='Visualization of the data of user logs of 8 users on Armsdealing data.', style={'marginTop': '-15px', 'marginBottom': '30px'})
    ], style={'textAlign': 'center'}),

    html.Div(children=[
    html.Div(children=[
            html.H2(id='freq', style={'fontWeight': 'bold', 'paddingTop': '.1rem', 'paddingLeft': '10rem'}),
            html.Label('Types of interactions performed', style={'paddingTop': '.3rem', 'paddingLeft': '5rem'}),
        ], className="three columns number-stat-box"),
    html.Div(children=[
            html.H2(id='docfreq', style={'fontWeight': 'bold', 'paddingTop': '.1rem', 'paddingLeft': '15rem'}),
            html.Label('Total Documents accessed', style={'paddingTop': '.3rem', 'paddingLeft': '10rem'}),
        ], className="three columns number-stat-box"),
    html.Div(children=[
            html.H2(id='time', style={'fontWeight': 'bold', 'paddingTop': '.1rem', 'paddingLeft': '15rem'}),
            html.Label('Total time considered in minutes', style={'paddingTop': '.3rem', 'paddingLeft': '10rem'}),
        ], className="three columns number-stat-box")
    ], style={'margin':'1rem', 'display': 'flex', 'justify-content': 'center', 'flex-wrap': 'wrap'}),



    
    dcc.RadioItems(id='userradio',
    options = [{'label': 'User 1', 'value': 'data_max_scaled'}, 
    {'label': 'User 2', 'value': 'data_max_scaled2'}, 
    {'label': 'User 3', 'value': 'data_max_scaled3'},
    {'label': 'User 4', 'value': 'data_max_scaled4'}, 
    {'label': 'User 5', 'value': 'data_max_scaled5'},
    {'label': 'User 6', 'value': 'data_max_scaled6'},
    {'label': 'User 7', 'value': 'data_max_scaled7'},
    {'label': 'User 8', 'value': 'data_max_scaled8'}], 
    value='data_max_scaled', style={'display': 'flex', 'justify-content': 'space-between', 'color': 'Black', 'font-size': 20, 
    'paddingTop': '5rem', 'width':'80%', 'paddingLeft': '8rem'}, 
    ),

    html.Div(children=[dcc.RangeSlider(
        int(data_max_scaled['time'].min()),
        int(data_max_scaled['time'].max()),
        step=10,
        id='crossfilter-year--slider',
        value=[data_max_scaled['time'].min(), data_max_scaled['time'].max()],
        
    )], style={'paddingTop': '4rem'}), 

    dcc.Graph(
            id='crossfilter-indicator-scatter',
            figure=fig,
            
        ),
    html.Img(id='image_wc', style={'display': 'inline-block', 'width': '30%', 'height': '40vw'}),
    
    dcc.Graph(
            id='rowbar',
            figure=figrow,
            style={'display': 'inline-block', 'width': '60%', 'height': '40vw'},

            
        ),
    

    

], )



@app.callback(
    [Output('freq', 'children'),Output('docfreq', 'children'),Output('time', 'children'), ],
    [Input('crossfilter-year--slider', 'value'),Input('userradio', 'value')]
    )
def display_value(value, value2):
    if(value2=='data_max_scaled'):
        value2=data_max_scaled
        
    elif(value2=='data_max_scaled2'):
        value2=data_max_scaled2
        
    elif(value2=='data_max_scaled3'):
        value2=data_max_scaled3
        
    elif(value2=='data_max_scaled4'):
        value2=data_max_scaled4
        
    elif(value2=='data_max_scaled5'):
        value2=data_max_scaled5
        
    elif(value2=='data_max_scaled6'):
        value2=data_max_scaled6
        
    elif(value2=='data_max_scaled7'):
        value2=data_max_scaled7
        
    elif(value2=='data_max_scaled8'):
        value2=data_max_scaled8
        
    dff = value2.loc[(value2["time"] >= value[0]) & (value2['time'] <=value[1])]
    x= list(dff['InteractionType'].unique())
    y= list(dff['ID'].unique())
    z= int(max(dff['time'])- min(dff['time']))

    return len(x), len(y), z
    


@app.callback(
    Output('rowbar', 'figure'),
    [Input('crossfilter-year--slider', 'value'),Input('userradio', 'value')]
    )
def plot_rowbar(value, value2):
    if(value2=='data_max_scaled'):
        value2=data_max_scaled
    elif(value2=='data_max_scaled2'):
        value2=data_max_scaled2
    elif(value2=='data_max_scaled3'):
        value2=data_max_scaled3
    elif(value2=='data_max_scaled4'):
        value2=data_max_scaled4
    elif(value2=='data_max_scaled5'):
        value2=data_max_scaled5
    elif(value2=='data_max_scaled6'):
        value2=data_max_scaled6
    elif(value2=='data_max_scaled7'):
        value2=data_max_scaled7
    elif(value2=='data_max_scaled8'):
        value2=data_max_scaled8

    
    
    dff = value2.loc[(value2["time"] >= value[0]) & (value2['time'] <=value[1])]
    d={}
    for i in range(0,len(dff)):
        if dff.iloc[i, 3] in d:
            d[dff.iloc[i, 3]]+= dff.iloc[i, 0]
        else:
            d[dff.iloc[i, 3]]= dff.iloc[i, 0]
    if '0' in d.keys():
        d.pop('0', 'None')
    d.pop('bottom-up', None)
    d.pop('top-down', None)
    # for f in d.keys():ß
    #     if ',' in f:
    #         d.pop(f)
    d1 = sorted(d.items(), key=lambda x:x[1], reverse=True)
    d = dict(d1)
    data={'documents':list(d.keys()), 'duration': list(d.values())}
    new = pd.DataFrame.from_dict(data)
    figrow = px.line_polar(new.loc[0:10, :], r='duration', theta='documents', line_close=True,
                        color_discrete_sequence=px.colors.sequential.Plasma_r)
    figrow.update_traces(fill='toself')
    figrow.update_traces(fillcolor="green", opacity=0.6, line=dict(color="green"))
    figrow.update_layout(
    template=None,
    polar = dict(radialaxis = dict(
        gridwidth=0.5,
                               
                              showticklabels=True, ticks='', gridcolor = "grey"),
                 angularaxis = dict(showticklabels=True, ticks='',
                               rotation=45,
                               direction = "clockwise",
                               gridcolor = "black"
                )))
    
    return figrow
    
        


@app.callback(
    Output('image_wc', 'src'),
    [Input('crossfilter-year--slider', 'value'),Input('userradio', 'value')]
    )
def plot_wordcloud(value, value2):
    dfdocs= pd.read_json('Documents_Dataset_1.json')
    if(value2=='data_max_scaled'):
        value2=data_max_scaled
    elif(value2=='data_max_scaled2'):
        value2=data_max_scaled2
    elif(value2=='data_max_scaled3'):
        value2=data_max_scaled3
    elif(value2=='data_max_scaled4'):
        value2=data_max_scaled4
    elif(value2=='data_max_scaled5'):
        value2=data_max_scaled5
    elif(value2=='data_max_scaled6'):
        value2=data_max_scaled6
    elif(value2=='data_max_scaled7'):
        value2=data_max_scaled7
    elif(value2=='data_max_scaled8'):
        value2=data_max_scaled8
       
    value2['ID']= [str(x).replace(' ', '') for x in value2['ID']]
    value2['ID']= value2['ID'].str.lower()
    dfdocs['ID']= dfdocs['ID'].str.lower()
    # print(df['ID'])
    dfcontents= pd.merge(value2, dfdocs, on='ID')
    dff = dfcontents.loc[(dfcontents["time"] >= value[0]) & (dfcontents['time'] <=value[1])]


    text=''
    dfwc=dff

    with open('readme.txt', 'w') as f:    
        for i in dfwc['title']:
            text=''.join(i)
            f.write(text)
            f.write(' ')


    text = open("readme.txt", mode="r").read()
    if '<br>' in text:
        text=text.replace('<br>', '')

    text=[text]

    vectorizer = TfidfVectorizer(stop_words='english')
    vecs = vectorizer.fit_transform(text)
    feature_names = vectorizer.get_feature_names()
    dense = vecs.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=feature_names)
    df.T.sum(axis=1)
    Cloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(df.T.sum(axis=1))
    img = BytesIO()
    Cloud.to_image().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    





@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    [Input('crossfilter-year--slider', 'value'),Input('userradio', 'value')]
    )



def updateScatter(value, value2):
    if(value2=='data_max_scaled'):
        value2=data_max_scaled
    elif(value2=='data_max_scaled2'):
        value2=data_max_scaled2
    elif(value2=='data_max_scaled3'):
        value2=data_max_scaled3
    elif(value2=='data_max_scaled4'):
        value2=data_max_scaled4
    elif(value2=='data_max_scaled5'):
        value2=data_max_scaled5
    elif(value2=='data_max_scaled6'):
        value2=data_max_scaled6
    elif(value2=='data_max_scaled7'):
        value2=data_max_scaled7
    elif(value2=='data_max_scaled8'):
        value2=data_max_scaled8

    dff = value2.loc[(value2["time"] >= value[0]) & (value2['time'] <=value[1])]
    # l=[]
    # l.append(min(dff['duration']))
    # l.append(sum(dff['duration'])//len(dff['duration']))
    # l.append(max(dff['duration']))
    # dff = value2[value2['time']>=(value[0])] + value2[value2['time']<=value[1]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(colorscale=[
        # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
        [0, "rgb(180, 180, 180)"],
        [0.34, "rgb(180,180,180)"],

        # Let values between 10-20% of the min and max of z
        # have color rgb(20, 20, 20)
        # [0.1, "rgb(160, 160, 160)"],
        # [0.3, "rgb(160, 160, 160)"],

        # Values between 20-30% of the min and max of z
        # have color rgb(40, 40, 40)
        # [0.2, "rgb(140, 140, 140)"],
        # [0.3, "rgb(140, 140, 140)"],

        # [0.3, "rgb(120, 120, 120)"],
        # [0.4, "rgb(120, 120, 120)"],

        # [0.4, "rgb(100, 100, 100)"],
        # [0.5, "rgb(100, 100, 100)"],

        [0.34, "rgb(70, 70, 70)"],
        [0.67, "rgb(70, 70, 70)"],

        # [0.6, "rgb(60, 60, 60)"],
        # [0.7, "rgb(60, 60, 60)"],

        # [0.7, "rgb(40, 40, 40)"],
        # [0.8, "rgb(40, 40, 40)"],

        # [0.8, "rgb(20, 20, 20)"],
        # [0.9, "rgb(20, 20, 20)"],

        [0.67, "rgb(0, 0, 0)"],
        [1.0, "rgb(0, 0, 0)"]
    ],

                   z=dff['duration'],
                   x=dff['time'],
                   y=dff['InteractionType'],
                   hoverongaps = False),
                   )
    fig.update_layout(
    title='Heatmap depicting and overview of analyst behavior over time',
    xaxis_title="Time (in minutes)",
    yaxis_title="Interaction Type",
    legend_title="Duration (in minutes)")

    return fig














if __name__ == '__main__':
    app.run_server(debug=True)