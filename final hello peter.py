import pandas as pd
from sklearn.model_selection import train_test_split
from io import StringIO
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import requests
import plotly.express as px
from nltk import sent_tokenize, word_tokenize
import dash
from dash import dcc, html, Input, State, Output, dash_table
import dash_bootstrap_components as dbc
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from string import punctuation
from collections import Counter
from heapq import nlargest
from nltk.tag import pos_tag
import re
import torch
from datetime import date
from spacy.lang.en.stop_words import STOP_WORDS

stp = list(STOP_WORDS)
bank_terms = ['standard', 'told', 'money','fnb', 'nedbank', 'Nedbank', 'Standard', 'Fnb', 'bank', 'capitec','yesterday','banks','week','weeks','year','years','months'
              'Capitec', 'absa','day','hello','peter','number','south','africa','days','Absa','std','the','standardbank','today','sunday','monday','tuesday','wednesday','thursday','friday','saturday']
stop_words = stp + bank_terms
pos_list = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBZ', 'VBD', 'VBG', 'VBN', 'VBP']

df_train = pd.read_excel("New Model.xlsx")

def date_to_week(input_date):
    year, week, _ = input_date.isocalendar()
    return pd.Series([year, week])


def clean_text(text):
    text = re.sub(r'XXXX', '', str(text))
    words_toks = word_tokenize(str(text))
    no_stop_words = [w for w in words_toks if
                     w not in stop_words and w not in punctuation and len(w) > 2 and w.isalpha()]
    nouns_verbs = [i for i, j in pos_tag(no_stop_words) if j in pos_list]
    fin_text = ' '.join(nouns_verbs)
    return fin_text


df_train['Cleaned'] = df_train['Review'].apply(clean_text)
df_train.dropna(inplace=True)



X = df_train['Cleaned']
y = df_train['Product']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=124)
pipeSVC = Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('clf',LinearSVC(dual=False))])
pipeSVC.fit(X_train,y_train)
predictSVC = pipeSVC.predict(X_test)

summarizer = pipeline("summarization",model='t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')





app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
server = app.server

def cat_review(col):
    if col < 3:
        cat = "Negative"
    elif col == 3:
        cat = "Neutral"
    else:
        cat = "Positive"
    return cat


bank_dict = {'standard-bank': 'Standard Bank', 'first-national-bank': 'Fnb', 'absa': 'Absa Bank', 'nedbank': 'Nedbank',
             'capitec-bank': 'Capitec Bank', 'african-bank': 'African Bank', 'tymebank': 'Tymebank',
             'discovery-bank': 'Discovery Bank',
             'investec': 'Investec Bank', 'rennies-foreign-exchange-bidvest-bank': 'Bidvest Bank'}

app.layout = html.Div([
    html.Div(
        [
            html.H5("Hellopeter: Sentiment Analysis on Customer Reviews",
                    style={"margin": "0", 'textAlign': 'center', "fontFamily": "Arial", "color": "white"}),
            html.P("This app focuses on the top ten banks in South Africa", style={"fontFamily": "Arial","margin": "0", "color": "white"}),
        ],
        style={
            "position": "fixed",
            "top": "0",
            "left": "0",
            "width": "100%",
            "backgroundColor": "#007bff",
            "color": "white",
            "padding": "10px",
            "zIndex": "1000",
            "textAlign": "center"
        },
    ),
    html.Div([
        html.P("App description", style={"color": "black", "padding": "10px", "font-weight": "bold","fontFamily": "Arial"})
    ], style={
        "position": "fixed",
        "top": "50px",  # Offset for the header
        "left": "0",
        "width": "200px",  # Sidebar width
        "height": "100vh",
        "backgroundColor": "lightgrey",
        "padding": "10px",
        "overflowY": "auto"
    }),
    html.Div([

    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H6("Select bank from the list", style={'fontFamily': 'Ariel'}),
            dcc.Dropdown(id='bank',
                         multi=False,
                         placeholder='Select the bank',
                         #value='standard-bank',
                         options=[{'label': 'Standard bank', 'value': 'standard-bank'},
                                  {'label': 'Fnb', 'value': 'first-national-bank'},
                                  {'label': 'Absa bank', 'value': 'absa'},
                                  {'label': 'Nedbank', 'value': 'nedbank'},
                                  {'label': 'Capitec bank', 'value': 'capitec-bank'},
                                  {'label': 'African bank', 'value': 'african-bank'},
                                  {'label': 'Tymebank', 'value': 'tymebank'},
                                  {'label': 'Discovery bank', 'value': 'discovery-bank'},
                                  {'label': 'Investec bank', 'value': 'investec'},
                                  {'label': 'Bidvest bank', 'value': 'rennies-foreign-exchange-bidvest-bank'}],style={'fontFamily':'Ariel','color':'black'})
        ]),
        dbc.Col([
            html.Div(id='maxim_page', style={'font-size': 15, 'color': 'lightgrey','fontFamily':'Ariel'}),
            dcc.Input(id='range', type='number', min=2, max=10, step=1,style={'font-size': 15, 'fontFamily':'Ariel'}),
        ], width=6),
        dbc.Col([
            html.Button('Download', id='download-button', n_clicks=0,
                        style={'font-size': 14, 'backgroundColor': 'lightgrey','fontFamily':'Ariel'})
        ]),
        dbc.Col([dcc.Store(id='dat')]),

    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id='loading-output',
                type='circle',
                children=[html.Div(id='data-description', style={'fontFamily': 'Ariel'})])
        ])
    ]),
    html.Hr(),
    dbc.Row([
        html.Div(html.Button("First 5 Rows of the Table - show/hide", id="toggle-button", n_clicks=0,style={'font-size': 14,'width':'20%','backgroundColor': 'lightgrey','fontFamily':'Ariel'}),style={'display': 'flex', 'justifyContent': 'center'}),
        html.Hr(),
        dash_table.DataTable(
            id='table',
            columns=[{"name": col, "id": col} for col in ["Date", "Review", "Ratings", "Bank", "Sentiment"]],
            # Default columns
            data=[],  # Empty at start
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
                        'overflow': 'hidden', 'textOverflow': 'ellipsis','fontFamily': 'Arial','fontSize': 13},
            style_data={'backgroundColor': 'black','color': 'white'},
            style_header={'backgroundColor': 'blue', 'fontWeight': 'bold','fontFamily': 'Arial','fontSize': 12},
            filter_action="native",
            sort_action="native",
            column_selectable="multi",
            row_selectable="multi",
            page_size=5)

    ]),
    html.Hr(),
    dbc.Row([
       html.Div(html.Button('Explore', id='explore-button', n_clicks=0,
                    style={'width': '15%', 'height': 40, 'font-size': 14, 'backgroundColor': 'lightgrey',
                           'fontFamily': 'Ariel'}),style={'display': 'flex', 'justifyContent': 'center'}),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Button(f"View {graph}-chart",id=f'view-{graph}',className='mb-3',color='primary',style={'width':'50%'}) for graph in ['pie','bar']
    ]),
    dbc.Row([
        dbc.Col(dbc.Collapse(
            dcc.Graph(id='pie-plot', style={'width': '80%', 'height': 300}),id='colapse-pie',is_open=False)
        ),
        dbc.Col(dbc.Collapse(
            dcc.Graph(id='bar-plot', style={'width': '85%', 'height': 300}),id='colapse-bar',is_open=False)
        ),


    ]),
    html.Hr(),
    dbc.Row([ html.H5('Summary of Analyzing Negative Customer Reviews',style={'textAlign':'center',"fontFamily":"Arial",'color':'darkgrey'}),
              html.Br(),
             dbc.Col([dcc.RadioItems(id='negative-sentiment',value='Negative',options=[{'label':'Negative sentiment','value':'Negative'}],style={'width': '30%'})]),
             dbc.Col([html.Button('Classify', id='classify-button', n_clicks=0,
                    style={'width': '30%', 'height': 40, 'font-size': 13, 'backgroundColor': 'lightgrey','fontFamily':'Ariel'})]),
             dbc.Col([dcc.Store(id='data2')]),
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
                     min_date_allowed='earliest_date',  # minimum date allowed on the DatePickerRange component
                     max_date_allowed='latest_date',  # maximum date allowed on the DatePickerRange component
                     # initial_visible_month=dt(2023,11,1),  # the month initially presented when the user opens the calendar
                     start_date='earliest_date',
                     end_date='latest_date',
                     display_format='DD-MM-YYYY',  # how selected dates are displayed in the DatePickerRange component.
                     updatemode='singledate',
                     style={'width': '30%'}

                 ),
             ]),
             dbc.Col([
                 dcc.Store(id='data3')
             ])


    ]),
    html.Hr(),
    dbc.Row([
        dbc.Button('View bar and line plot', id='view-bar-line-chart', className='mb-3', color='primary')
    ], justify='center'),
    html.Hr(),
    dbc.Row([
        dbc.Collapse(
            dbc.Row([
                dbc.Col([
                    # Horizontal bar chart
                    dcc.Graph(id='hbar', style={'height': 300})
                ], width=4),  # Adjust width for layout

                dbc.Col([
                    # Radio items for selecting options
                    dcc.RadioItems(
                        id='clist',
                        options=[],
                        style={'fontFamily': 'Arial', 'height': 40, 'font-size': 13}
                    )
                ], width=2),  # Adjust width for layout

                dbc.Col([
                    # Trend line plot
                    dcc.Graph(id='trend-plot', style={'height': 300})
                ], width=4)  # Adjust width for layout
            ]),
            id='bar-line-chart',
            is_open=False
        )
    ]),
    html.Hr(),
    dbc.Row([html.H5('Extractive and Abstractive Summarization of Negative Reviews',style={'textAlign':'center',"fontFamily":"Arial",'color':'darkgrey'}),
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
                dcc.Loading(
                    id='load-abstract',
                    type='circle',
                    children=[
                        dcc.Textarea(
                            id='abstract',
                            placeholder='Abstractive summarization will appear here...',
                            style={'width': '100%', 'height': '150px', 'font-size': '13px'}
                        )
                    ]
                )
            ])

        ])


]),
],style={
        "marginLeft": "220px",  # Leave space for the sidebar
        "marginTop": "60px",  # Leave space for the header
        "padding": "20px",
        "overflowY": "auto"
    }),
])

# Update numpage options based on bank selection
@app.callback(
    Output('maxim_page', 'children'),
    Output('range', 'max'),  # Dynamically update max value
    Input('bank', 'value')
)
def update_page_range(bank):
    if bank:
        url = f'https://api.hellopeter.com/consumer/business/{bank}/reviews'
        try:
            response = requests.get(url)
            response.raise_for_status()
            init_dictionary = response.json()
            last_page = init_dictionary.get('last_page', 10)  # Default max 10 if missing
            return f'Enter not more than {last_page} pages.', last_page
        except requests.exceptions.RequestException as e:
            return f"Error fetching data: {str(e)}", 10  # Default max to 10 in case of an error
    return "", 10  # Default empty response and max value


# Process and fetch data with progress bar update
@app.callback(Output('dat', 'data'),
              Input('download-button', 'n_clicks'),
              State('bank', 'value'),
              State('range', 'value'))
def bank_page(n_clicks, bank, pages):
    if n_clicks is not None and n_clicks > 0 and bank and pages:
        url = f'https://api.hellopeter.com/consumer/business/{bank}/reviews'
        response = requests.get(url)
        init_dictionary = response.json()
        hp_dictionary_allpages = []
        for page in range(pages):

            ur = init_dictionary['first_page_url'][:-1] + str((page + 1))
            response2 = requests.get(ur)

            # Parsing Data as Json Dictionary and appending it tou our hp_dictionary_allpages list
            try:
                data = response2.json()
                hp_dictionary_allpages.append(data)
            except Exception as e:
                data = None
                hp_dictionary_allpages.append(data)
            rates = []
            review = []
            date = []
            bankn = []
            for i in range(len(hp_dictionary_allpages)):
                for j in range(11):
                    try:
                        rev = hp_dictionary_allpages[i]['data'][j]['review_content']
                        review.append(rev)
                        dat = hp_dictionary_allpages[i]['data'][j]['created_at']
                        date.append(dat)
                        rating = hp_dictionary_allpages[i]['data'][j]['review_rating']
                        rates.append(rating)
                        bankn.append(bank)
                    except Exception as e:
                        rating = None
                        rates.append(rating)
                        rev = None
                        review.append(rev)
                        dat = None
                        date.append(dat)
                        bankn.append(bank)
        df = pd.DataFrame({'Date': date, 'Review': review, 'Ratings': rates, 'Bank': bankn})
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df.dropna(inplace=True)
        df['Sentiment'] = df['Ratings'].apply(cat_review)
        df.to_excel(fr'C:\Users\mesul\Documents\Python Scripts\bank_{bank}.xlsx')
        return df.to_json(date_format='iso', orient='split')
    return None


@app.callback(Output('data-description', 'children'),
              Input('download-button', 'n_clicks'),
              State('dat', 'data'))
def data_describe(cliq, dat):
    if cliq is not None and cliq > 0 and dat:
        time.sleep(2)
        df = pd.read_json(StringIO(dat), orient='split')
        cols = list(df.columns)
        col = df.shape[1]
        rows = df.shape[0]
        name = bank_dict[df['Bank'].unique()[0]]
        return f'The downloaded data table of the {name} consists of {col} columns: namely {cols[0]}, {cols[1]}, {cols[2]}, {cols[3]} and {cols[4]}. The total number of customer reviews is {rows}.'
    return ''


@app.callback(
    Output('bar-plot', 'figure'),
    [Input('explore-button', 'n_clicks')],
    [State('dat', 'data')]
)
def explore_bar(cliq, dat):
    if cliq is not None and dat:
        try:
            # Convert JSON data from Dash `State` into a DataFrame
            df_ = pd.read_json(StringIO(dat), orient='split')

            # Ensure 'Date' and 'Sentiment' columns exist
            if 'Date' not in df_ or 'Sentiment' not in df_:
                raise ValueError("Data is missing required columns 'Date' or 'Sentiment'.")

            # Extract year, month, and week information
            # df_['Month'] = pd.to_datetime(df_['Date']).dt.month
            # df_['Year'] = pd.to_datetime(df_['Date']).dt.year
            # df_['Week'] = pd.to_datetime(df_['Date']).dt.isocalendar().week
            df_[['year', 'week']] = df_['Date'].apply(date_to_week)
            df_['dateweek'] = df_.apply(lambda x: date.fromisocalendar(x['year'], x['week'], 1), axis=1)
            #df_['dateweek'] = df_.apply(lambda row: week_to_date(row['Week'], row['Year']), axis=1)



            # Group data by the calculated week date and sentiment
            df_group = df_.groupby(['dateweek', 'Sentiment']).size().reset_index()
            df_group.columns = ['date', 'Sentiment', 'Count']
            df_group.sort_values(by='date', inplace=True)


            # Create bar plot
            figg = px.bar(df_group, x='date', y='Count', color='Sentiment')

            figg.update_layout(
                title='Number of Reviews by Sentiment Type Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Reviews',
                template='plotly_white',
                font_family='Arial'
            )
            return figg

        except Exception as e:
            # Log the exception and return an empty figure
            print(f"Error processing data: {e}")
            return px.bar()  # Return an empty bar chart if there’s an error

    # Return an empty figure if `cliq` is None or `dat` is empty
    return px.bar()


@app.callback(Output('pie-plot', 'figure'),
              [Input('explore-button', 'n_clicks')],
              [State('dat', 'data')])
def explore_pie(cliq, dat):
    if cliq and dat:
        df = pd.read_json(StringIO(dat), orient='split')
        senti_group = df['Sentiment'].value_counts().reset_index()
        senti_group.columns = ['Sentiment', 'count']
        #name = bank_dict[df['Bank'].unique()[0]]
        name = bank_dict.get(df['Bank'].iat[0], 'Unknown Bank')
        fig = px.pie(senti_group, values='count', names='Sentiment', hole=.3)

        # Customize layout
        fig.update_layout(
            title=f' Customer Review Sentiment Analysis for {name}',
            font_family='Ariel'
        )

        return fig
    return {}


@app.callback(
    Output('data2', 'data'),
    Output('daterange', 'start_date'),
    Output('daterange', 'end_date'),
    Input('classify-button', 'n_clicks'),
    State('dat', 'data'),
    State('negative-sentiment', 'value')
)
def filtered_data(cliq, dat, sent):
    if cliq and dat and sent:
        try:
            # Convert JSON data to DataFrame
            dff = pd.read_json(StringIO(dat), orient='split')

            # Filter data based on sentiment and make a copy to avoid SettingWithCopyWarning
            dff1 = dff[dff['Sentiment'] == sent].copy()

            # Ensure 'Date' column is in datetime format
            dff1['Date'] = pd.to_datetime(dff1['Date'], errors='coerce')
            dff1.dropna(subset=['Date'], inplace=True)  # Drop rows where 'Date' conversion failed

            # Predict 'Product' category for each review
            dff1['Product'] = dff1['Review'].apply(lambda x: pipeSVC.predict([str(x)])[0])

            # Calculate earliest and latest dates
            earliest_date = dff1['Date'].min().date()
            latest_date = dff1['Date'].max().date()

            # Convert filtered data back to JSON format for Dash
            return dff1.to_json(date_format='iso', orient='split'), earliest_date, latest_date

        except Exception as e:
            print(f"Error in filtered_data callback: {e}")
            return None, None, None

    # If conditions are not met, return None
    return None, None, None


@app.callback(
    Output('hbar', 'figure'),
    Output('clist', 'options'),
    Output('data3', 'data'),
    Input('data2', 'data'),
    Input('daterange', 'start_date'),
    Input('daterange', 'end_date')
)
def hbar_chart(datt, startsd, endsd):
    try:
        if not datt or not startsd or not endsd:
            return px.bar(), [], None

        # Load data and filter by date range
        dff = pd.read_json(StringIO(datt), orient='split')
        dff['Date'] = pd.to_datetime(dff['Date'], errors='coerce')  # Ensure Date is in datetime format
        dfff = dff[(dff['Date'] >= startsd) & (dff['Date'] <= endsd)].copy()

        # Verify filtered data is not empty
        if dfff.empty:
            return px.bar(), [], None

        # Group data by 'Product' and get the top 5 by count
        dff1 = dfff.groupby('Product').size().reset_index(name='Total').sort_values(by='Total', ascending=False)
        df_top5 = dff1.head(5)

        # Create list of options for dropdown
        prd_list = [{'label': product, 'value': product} for product in df_top5['Product']]

        # Create horizontal bar chart
        figi = px.bar(df_top5.sort_values(by='Total'), x='Total', y='Product', text='Total',
                      title='Top Five Issues',
                      labels={'Total': 'Count', 'Product': ''})
        figi.update_traces(textposition='inside', textfont_size=9)
        figi.update_layout(title_x=0.5, title_font=dict(size=15),
                           template="simple_white", font_family='Arial')

        # Convert filtered data to JSON format
        return figi, prd_list, dfff.to_json(date_format='iso', orient='split')

    except Exception as e:
        print(f"Error in hbar_chart callback: {e}")
        return px.bar(), [], None


@app.callback(Output('trend-plot','figure'),
              Input('data3','data'),
              Input('clist','value'))

def plot_trend(dat,clis):
    if dat and clis:
        df = pd.read_json(StringIO(dat),orient='split')
        df1 = df[df['Product']==clis]
        # df1['Year'] = pd.to_datetime(df1['Date']).dt.year
        # df1['Week'] = pd.to_datetime(df1['Date']).dt.isocalendar().week
        df1[['year', 'week']] = df1['Date'].apply(date_to_week)
        df1['dateweek'] = df1.apply(lambda x: date.fromisocalendar(x['year'], x['week'], 1), axis=1)
        #df1['dateweek'] = df1.apply(lambda row: week_to_date(row["Week"], row["Year"]), axis=1)
        df2 = df1.groupby('dateweek').size().reset_index()
        df2.columns = ['Date','Count']
        fig = px.line(df2,x='Date',y='Count',labels={'Count':'Number'})
        fig.update_layout(title_text=f'Number of issues concerning {clis}', title_x=0.5,font_family='Arial' ,title_font=dict(size=15),
                          template="simple_white", margin=dict(l=10, r=10, t=30, b=20))

        return fig
    return {}

@app.callback(
    Output('outext', 'value'),
    [Input('data3', 'data'), Input('clist', 'value')]
)
def text_area(data, mlist):
    try:
        # Convert JSON data to DataFrame and filter based on selected product
        df = pd.read_json(StringIO(data), orient='split')
        col = df[df['Product'] == mlist]['Review']

        # Concatenate all reviews into one large text string
        nlp_text = col.str.cat(sep=" ")
        if not nlp_text:
            return "No text data available."

        # Tokenize sentences and process each word for keyword frequency
        lis_word = [
            word.lower() for sent in sent_tokenize(nlp_text)
            for word in word_tokenize(sent)
            if word.lower() not in stop_words and word not in punctuation and len(word) >= 3 and word.isalpha()
        ]
        keywordz_freq = Counter(lis_word)

        def remove_emails_and_urls(text):
            # Remove email addresses
            text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '', text)
            # Remove URLs (http, https, www)
            text = re.sub(r'(https?://\S+|www\.\S+)', '', text)

            # Clean up extra whitespace
            cleaned_text = ' '.join(text.split())

            # Split into sentences using '.', '!', or '?' as delimiters
            sentences = re.split(r'[.!?]', cleaned_text)

            # Filter and keep only sentences with more than 2 words
            filtered = [s.strip() for s in sentences if len(s.split()) > 2]

            # Join filtered sentences into one paragraph
            para = '. '.join(filtered)
            return para

        nlp_text1 = remove_emails_and_urls(nlp_text)

        # Calculate sentence strength if keywords are found
        if keywordz_freq:
            sentence_strength = {}
            for sent in sent_tokenize(nlp_text1):
                for word in word_tokenize(sent):
                    if word.lower() in keywordz_freq:
                        sentence_strength[sent] = sentence_strength.get(sent, 0) + keywordz_freq[word.lower()]

            # Get top 5 sentences based on strength
            top5 = nlargest(5, sentence_strength, key=sentence_strength.get)

            # Function to clean spaces within sentences
            def clean_space(txt):
                return [re.sub(r'\s+', ' ', sent).strip() for sent in txt]

            # Clean and format sentences for output
            clean_list = clean_space(top5)
            fin_sentence = [sent for i, sent in enumerate(clean_list)]
        else:
            fin_sentence = ["No significant keywords found."]

        # Join sentences with line breaks for display
        return '\n\n'.join(fin_sentence)

    except Exception as e:
        print(f"Error in text_area callback: {e}")
        return "An error occurred while processing the data."



@app.callback(
    Output('abstract', 'value'),
    [Input("submit-button", "n_clicks")],
    [State('outext', 'value')]
)
def abstract_summa(n_clicks, text):
    if n_clicks is not None and n_clicks > 0 and text:
        summa_list = []
        try:
            for sent in sent_tokenize(text):
                preprocessed_text = sent.strip().replace('\n', '')
                t5_input_text = 'summarize: ' + preprocessed_text
                tokenized_text = tokenizer.encode(t5_input_text,
                                                  return_tensors='pt', truncation=True).to(device)
                summary_ids = model.generate(tokenized_text, min_length=15, max_length=30)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Append each summary to the list
                summa_list.append(summary)


            # Join all summarized sentences into one output
            summary1 = '  '.join(summa_list)
            summary = ' '.join(sent.capitalize() for sent in sent_tokenize(summary1))

        except Exception as e:
            print(f"Error: {e}")
            summary = "Error occurred while summarizing the text."
    else:
        summary = "Abstractive summarization will appear here..."

    # Return the summary as the output value for the 'abstract' component
    return summary




@app.callback(
    [Output('colapse-pie', 'is_open'), Output('colapse-bar', 'is_open')],
    [Input('view-pie', 'n_clicks'), Input('view-bar', 'n_clicks')],
    [State('colapse-pie', 'is_open'), State('colapse-bar', 'is_open')]
)
def toggle_collapse(n_clicks_pie, n_clicks_bar, is_open_pie, is_open_bar):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open_pie, is_open_bar
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'view-pie':
        return not is_open_pie, is_open_bar
    elif button_id == 'view-bar':
        return is_open_pie, not is_open_bar
    return is_open_pie, is_open_bar


@app.callback(
    Output("bar-line-chart", "is_open"),
    [Input("view-bar-line-chart", "n_clicks")],
    [State("bar-line-chart", "is_open")]
)
def toggle_collapse(n, is_open):
    # Check if the button was clicked (n is not None)
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('table', 'data'),
    Input("toggle-button", "n_clicks"),
    State('dat', 'data')
)

def update_table(n_clicks, stored_data):
    if n_clicks is None:  # Handle initial load
        return []

    if n_clicks % 2 == 0 and stored_data:
        df = pd.read_json(stored_data, orient='split')  # Convert JSON to DataFrame
        return (df.to_dict("records"))  # Show only the first 7 rows

    return []

if __name__ == '__main__':
    app.run_server(debug=True)
