import os
import pandas as pd
import numpy as np
import dash
import re
import openpyxl
from datetime import datetime as dt
from dash import dcc,html,Input,State,Output,Dash
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.express as px
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, pipeline
from io import StringIO
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize

os.chdir(r"C:\Users\mesul\Documents\Python Scripts")



df = pd.read_excel('class.xlsx')
df_piv = df.groupby(['Date1', 'Bank', 'Sentiment']).size().reset_index().sort_values(by='Date1')
df_piv_w = pd.pivot_table(df_piv, index=['Date1', 'Bank'], columns='Sentiment', values=0, aggfunc='sum').reset_index()
df_class = df
df_piv_w['% of Disappointment'] = round(
    (df_piv_w['Disappointed'] / df_piv_w[['Disappointed', 'Happy', 'Neutral']].sum(axis=1)) * 100, 0)

stop_words = list(STOP_WORDS)
stop_words.remove('not')
term_list = ['account', 'bank', 'Account', 'Bank']
stop_words = stop_words + term_list

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE], suppress_callback_exceptions=True)
server = app.server

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    children=[
        html.H4(children="Taps bar", className="display-7"),
        html.Hr(),
        html.P(
            children="Top 5 Banks of SA: Customer Reviews", className="lead"
        ),
        dbc.Nav(
            children=[
                dbc.NavLink(children="Sentiments Overview", href="/", active="exact"),
                dbc.NavLink(children="Top 10 negative reviews", href="/page-1", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/page-1":
        return [
            html.H5('Summary of Analyzing Negative Customer Reviews Across Big 5 Banks',
                    style=dict(textAlign='center', color='darkgrey', font_family="Arial")),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(id='bank',
                                 multi=False,
                                 value='nedbank',
                                 options=[{'label': 'STD bank', 'value': 'standard-bank'},
                                          {'label': 'FNB', 'value': 'first-national-bank'},
                                          {'label': 'Absa', 'value': 'absa'},
                                          {'label': 'Nedbank', 'value': 'nedbank'},
                                          {'label': 'Capitec', 'value': 'capitec-bank'}]),

                ]),
                dbc.Col([
                    dcc.RadioItems(id='senti',
                                   value='Disappointed',
                                   options=[{'label': 'Negative review', 'value': 'Disappointed'}])
                ]),
                dbc.Col([
                    dcc.DatePickerRange(
                        id='daterange',
                        calendar_orientation='horizontal',  # vertical or horizontal
                        day_size=39,  # size of calendar image. Default is 39
                        end_date_placeholder_text="Return",  # text that appears when no end date chosen
                        with_portal=True,  # if True calendar will open in a full screen overlay portal
                        first_day_of_week=0,  # Display of calendar when open (0 = Sunday)
                        reopen_calendar_on_clear=False,
                        is_RTL=False,  # True or False for direction of calendar
                        clearable=False,  # whether or not the user can clear the dropdown
                        number_of_months_shown=1,  # number of months shown when calendar is open
                        min_date_allowed='mindate',  # minimum date allowed on the DatePickerRange component
                        max_date_allowed='maxdate',  # maximum date allowed on the DatePickerRange component
                        # initial_visible_month=dt(2023,11,1),  # the month initially presented when the user opens the calendar
                        start_date='mindate',
                        end_date='maxdate',
                        display_format='MMM Do YY',
                        # how selected dates are displayed in the DatePickerRange component.
                        updatemode='singledate'

                    )
                ]),
                dbc.Col([
                    dcc.Store(id='prod')
                ]),
                dbc.Col([
                    dcc.Store(id='dat')
                ]),

            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='hbar')
                ]),
                dbc.Col([
                    dcc.Graph(id='tren')
                ]),
                dbc.Col([
                    dcc.RadioItems(id='clist',
                                   options=[],
                                   value='prod', labelStyle={'display': 'inline-block', 'margin-right': '10px'})
                ])

            ]),
            dbc.Row([
                dbc.Col([html.Br(),
                         html.H6("Use extractive summarization to obtain the essence of the top 5 reviews"),
                         html.Div([
                             dcc.Textarea(id='outext',
                                          placeholder='Choose a single option from the radio bttuon',
                                          style={'width': '100%', 'height': 300, 'font-size': 13}, )
                         ])
                         ]),
                dbc.Col([
                    html.Br(),
                    html.Button(children='Summarize', id='submit-button', n_clicks=0,
                                style={'font-size': 13, 'backgroundColor': 'pink'}),
                    html.Br(),
                    html.Br(),
                    html.H6("Abstractive summarization"),
                    html.Div([
                        dcc.Textarea(id='abstract',
                                     placeholder='Abstractive summarization',
                                     style={'width': '100%', 'height': 150, 'font-size': 13}, )
                    ])
                ])
            ]),

        ]



    elif pathname == "/":
        return [
            html.H5(children='Sentiment Analysis on Customer Reviews', style=dict(textAlign='center',
                                                                                  font_family='Arial',color='darkgrey')),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(id='banks',
                                 multi=False,
                                 value='nedbank',
                                 options=[{'label': 'STD bank', 'value': 'standard-bank'},
                                          {'label': 'FNB', 'value': 'first-national-bank'},
                                          {'label': 'Absa', 'value': 'absa'},
                                          {'label': 'Nedbank', 'value': 'nedbank'},
                                          {'label': 'Capitec', 'value': 'capitec-bank'}], style={'width': '60%'}),
                    dcc.Graph(id='line'),
                    dcc.Graph(figure=fig)], )

            ]),
        ]

    return dbc.Jumbotron(
        [
            html.H4(children="404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


dff = df_piv_w.copy()
fig = px.line(dff, x='Date1', y='% of Disappointment',
              title='Monthly Proportion of Negative Reviews', color='Bank',
              labels={'variable': 'Sentiment', 'value': '%', 'Date1': 'Date', '% of Disappointment': '%',
                      'Bank': 'Top 5 banks'})
fig.update_traces(mode='lines+markers')
fig.update_layout(paper_bgcolor='white')
fig.update_traces(line=dict(dash="dot", width=2))
# fig.update_layout(plot_bgcolor="lightgrey")
fig.update_layout(
    plot_bgcolor='white',  # "simple_white",
    width=1000, title_x=0.5,
    height=400,
    font_family='Arial'
)
fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
)
fig.update_yaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
)


@app.callback(Output('line', 'figure'),
              Input('banks', 'value')
              )
def chart(dropval):
    df_line = df_piv_w.copy()
    figu = px.line(df_line[df_line.Bank.isin([dropval])], x='Date1', y=['Disappointed', 'Neutral', 'Happy'],
                   title=f"Overview of Monthly Sentiments for: {dropval.capitalize()}",
                   labels={'variable': 'Sentiment', 'value': 'Number', 'Date1': 'Date'})
    figu.update_traces(mode='lines+markers')
    figu.update_layout(paper_bgcolor='white')
    figu.update_traces(line=dict(dash="dot", width=2))
    figu.update_layout(
        plot_bgcolor='white', title_x=0.5,
        width=1000,
        height=400,
        font_family='Arial'
    )
    figu.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    figu.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    # figu.update_layout(plot_bgcolor="lightpink")
    return figu


@app.callback(Output('daterange', 'start_date'),
              Output('daterange', 'end_date'),
              Input('bank', 'value'),
              Input('senti', 'value'))
def calender(drp, senti):
    df = df_class.copy()
    df1 = df[(df.Bank.isin([drp])) & (df.Sentiment.isin([senti]))]
    mindate = min(df1['Date1']).date()
    maxdate = max(df1['Date1']).date()
    return mindate, maxdate


@app.callback(Output('hbar', 'figure'),
              Output('prod', 'value'),
              Output('dat', 'data'),
              Output('clist', 'options'),
              Input('bank', 'value'),
              Input('senti', 'value'),
              Input('daterange', 'start_date'),
              Input('daterange', 'end_date'))
def hbar_chart(drp, senti, startdate, enddate):
    df_bar = df_class.copy()
    dff = df_bar[(df_bar.Bank.isin([drp])) & (df_bar.Sentiment.isin([senti]))]
    dfff = dff[(dff.Date1 >= startdate) & (dff.Date1 < enddate)]
    dff0 = dfff.copy()

    dff1 = dff0.groupby('Product').size().reset_index().sort_values(by=0, ascending=False)[:10]
    dff1.columns = ['Product', 'Total']
    uni_list = list(dff1['Product'].unique())
    prd_list = [{'label': category, 'value': category} for category in dff1['Product'].unique()]

    df_p = dff0[dff0.Product.isin(uni_list)]

    dff1 = dff1.sort_values(by='Total')
    top = dff1.tail(1)
    prod = top['Product'].to_list()[0]

    figi = px.bar(dff1, x='Total', y='Product', text='Total',
                  title=f'Top 10 issues related to {drp.capitalize()} ',
                  labels={'Total': '', 'Product': ''}, height=300, width=440)
    figi.update_traces(textposition='inside', textfont_size=9)
    figi.update_layout(title_x=0.5, title_font=dict(size=15), margin=dict(l=10, r=10, t=30, b=20),
                       template="simple_white",font_family='Arial')
    return figi, prod, df_p.to_json(date_format='iso', orient='split'), prd_list


@app.callback(Output('tren', 'figure'),
              Input('dat', 'data'),
              Input('prod', 'value'),
              Input('clist', 'value')
              )
def check_data(datt, col, clis):
    df = pd.read_json(StringIO(datt), orient='split')
    df1 = df[df['Product'] == col].groupby('Date1').size().reset_index().sort_values(by='Date1')
    fi3 = px.line(df1, x='Date1', y=0, labels={'0': 'Number', 'Date1': 'Date'}, height=300, width=440)
    fi3.update_layout(title_text=f'Number of issues concerning {col}', title_x=0.5, title_font=dict(size=15),
                      margin=dict(l=10, r=10, t=30, b=20))
    if clis:
        df1 = df[df['Product'].isin([clis])].groupby(['Date1', 'Product']).size().reset_index().sort_values(by='Date1')

        fi3 = px.line(df1, x='Date1', y=0,
                      labels={'0': 'Number', 'Date1': 'Date'}, height=300, width=440)
        fi3.update_layout(title_text=f'Number of issues concerning {clis}', title_x=0.5,font_family='Arial' ,title_font=dict(size=15),
                          template="simple_white", margin=dict(l=10, r=10, t=30, b=20))

    return fi3


@app.callback(Output('outext', 'value'),
              Input('dat', 'data'),
              Input('clist', 'value'))
def text_area(data, mlist):
    df = pd.read_json(StringIO(data), orient='split')
    col = df[df['Product'] == mlist]['Review']
    nlp_text = col.str.cat(sep=" ")
    nlp_docs = sent_tokenize(nlp_text)
    lis_word = []
    for sent in sent_tokenize(nlp_text):
        for word in word_tokenize(sent):
            if word.lower() not in stop_words and word not in punctuation and len(word) >= 3 and word.isalpha():
                lis_word.append(word)
    keywordz_freq = Counter(lis_word)

    if keywordz_freq:
        key = keywordz_freq.most_common(1)[0][1]
        for item in keywordz_freq.keys():
            keywordz_freq[item] = (keywordz_freq[item] / key)

        sentence_strength = {}
        for sent in sent_tokenize(nlp_text):
            for word in word_tokenize(sent):
                if word in keywordz_freq.keys():
                    if sent in sentence_strength.keys():
                        sentence_strength[sent] += keywordz_freq[word]
                    else:
                        sentence_strength[sent] = keywordz_freq[word]

        top5 = nlargest(5, sentence_strength, key=sentence_strength.get)
        fin_sentence = [str(i + 1) + ". " + str(sent).strip()
                        for i, sent in enumerate(top5)]

    else:
        fin_sentence = []

    return ' '.join(fin_sentence)

@app.callback(
    Output('abstract', 'value'),
    [Input("submit-button", "n_clicks")],
    [State('outext', 'value')]
)
def abstract_summa(n_clicks, text):
    if n_clicks is not None and n_clicks > 0 and text:
        try:
            t5_summarizer1 = pipeline("summarization", model="t5-large")
            bart1 = t5_summarizer1(text, max_length=250, min_length=50, do_sample=False)[0]['summary_text']
            summary = bart1
        except Exception as e:
            print(f"Error: {e}")
            summary = "Error occurred while summarizing the text."
    else:
        summary = ""

    return summary



if __name__ == '__main__':
    app.run_server(debug=True)
