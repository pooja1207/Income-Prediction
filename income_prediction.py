# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:00:28 2020

@author: pooja
"""


import streamlit as st

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_loc = "adult.csv"

@st.cache
def load_data(dataset_loc):
    missing_values = ['?']
    df = pd.read_csv(dataset_loc,na_values = missing_values)
    return df

def load_intro(df):
    st.header('Introduction ')
    st.markdown('''The Adult Data Set contains the record of the workers. The dataset is credited to Ronny Kohavi and Barry Becker and was drawn from the 1994 United States Census Bureau data. The target variable of this dataset is Income. On given attributes, the aim is to predict the annual income of a individual whether it is less than, greater than or equal to 50K.\n 
- There are a total of 48,842 rows and 15 attributes including target attribute(income).
- The dataset contains missing values that are marked with a question mark character (?).
- The given dataset has both categorical and numeric variables.
- There are two class values ‘>50K‘ and ‘<=50K‘, meaning it is a binary classification task. 
     - '>50K' : majority class, approximately 25%.
     - '<=50K': minority class, approximately 75%.''')
     
    st.subheader('What I got to know about the features in dataset')
    st.markdown('''- age of the worker (continuous and numerical)<br>
- workclass tells the class of work (categorial and non-numerical) :- Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked<br>
- fnlwgt tells the final weight of how much of the population it represents (continuous and numerical)<br>
- education (categorial and non-numerical) :- Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool<br>
- educational-num tells the numeric education level (continuous and numerical)<br>
- marital-status (categorical and non-numeric) :- Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse <br>
- occupation (categorical and non-numeric) :- Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces <br>
- relationship (categorical and non-numeric) :- Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried <br>
- race (categorical and non-numerical) :- White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black <br>
- sex (categorical and non-numerical) :- Female, Male <br>
- capital-gain (continuous and numerical) <br>
- capital-loss (continuous and numerical) <br>
- hours-per-week shows the average number of hour working per week of an individual (continuous and numerical) <br>
- native-country (categorical and non-numeric) :- United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands <br>
- Income shows the annual income of workers (discrete) :- >50K, <=50K''')

    st.subheader('Objective')
    st.markdown('''To perform the exploratory data analysis on Adult dataset and to predict the annual income of the workers based on their given 14 attributes. Annual income is our class which is to be predicted for new data point and can have two different values : >50K or <=50K.''')


def load_description(df):
    # Preview of the dataset
        st.header("Data Preview")
        preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
        if(preview == "Top"):
            st.write(df.head())
        if(preview == "Bottom"):
            st.write(df.tail())

        # display the whole dataset
        if(st.checkbox("Show complete Dataset")):
            st.write(df)
            
        if(st.checkbox("Show data description")):
            st.write(df.describe(include = "all"))

        # Show shape
        if(st.checkbox("Display the shape")):
            st.write(df.shape)
            dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
            if(dim == "Rows"):
                st.write("Number of Rows", df.shape[0])
            if(dim == "Columns"):
                st.write("Number of Columns", df.shape[1])

        # show columns
        if(st.checkbox("Show the Columns")):
            st.write(df.columns)
            
def load_missing_values(df):
    st.header('Missing Value Treatment')
    if(st.checkbox('Show the no. of missing value in each column')):
        st.write(df.isnull().sum())
        st.markdown('''Observation
- There are 2799, 2809 and 857 missing values in workclass, occupation and native-country respectively.''')
    if(st.checkbox('Show the missing value pattern')):
        # Pattern of missing values
        plt.figure(figsize=(10,5))
        sns.heatmap(df.isnull(),cbar=True,cmap="PuBuGn",cbar_kws={'label': 'Colorbar'})
        st.pyplot()
        st.markdown('''Observation
- There are 7% missing values.
- There are missing values in three columns : workclass, occupation and native-country.
- There is no relationship between the missingness of the native-region and any other features. Those missing data points of native-region are a random subset of the data. Therefore, the missing values of native-region are Missing Completely at Random (MCAR).
- When the workclass of a worker is missing, his occupation is also missing but vice-versa is not true. Therefore, we can conclude that missingness of workclass is dependent on missingness of occupation. Therefore, the missing values of workclass and occupation are following the Missing at Random (MAR) pattern.''')           
    
    # Handling Missing Values
    # Removing all the missing values by removing the rows where missing values occur
    new_df=df.dropna()
    
    # Removing the ‘fnlwgt’ column  as it has no predictive power
    new_df = new_df.drop(columns = "fnlwgt")
    
    if(st.checkbox('Data after missing value and unneccessary column treatment')):
        st.write(new_df.head())
    
    if(st.checkbox('Percentage Loss of data during treatment')):
        fraction = 1-(len(new_df.index)/len(df.index))
        perc = fraction*100
        st.write("Percentage of removed rows : {:.2f}%".format(perc))

def load_uni_num(df):
    st.subheader('Univariate Analysis of Numerical Columns')
    df=df.dropna()
    df = df.drop(columns = "fnlwgt")
    if(st.checkbox('Age')):
        # Age
        plt.figure(figsize=(15,5))
        
        plt.subplot(1,3,1)
        plt.title("Box Plot - Age")
        plt.boxplot(df['age'])
        
        plt.subplot(1,3,2)
        plt.title("Histographs - Age")
        plt.hist(df['age'])
        
        plt.subplot(1,3,3)
        plt.title("PDF - Age")
        sns.distplot(df['age'],hist=True, rug=False)
        st.pyplot()
        
    if(st.checkbox('Educational-Num')):
        # Educational-num

        plt.figure(figsize=(15,5))
        
        plt.subplot(1,3,1)
        plt.title("Box Plot - Educational-num")
        plt.boxplot(df['educational-num'])
        
        plt.subplot(1,3,2)
        plt.title("Histographs - Educational-num")
        plt.hist(df['educational-num'])
        
        plt.subplot(1,3,3)
        plt.title("PDF - Educational-num")
        sns.distplot(df['educational-num'],hist=False, rug=True)
        st.pyplot()
      
    if(st.checkbox('Capital-Gain')):
        # Capital-gain

        plt.figure(figsize=(15,5))
        
        plt.subplot(1,3,1)
        plt.boxplot(df['capital-gain'])
        plt.title("Box Plot - Capital-Gain")
        
        plt.subplot(1,3,2)
        plt.hist(df['capital-gain'],histtype='step')
        plt.title("Histograph - Capital-Gain")
        
        plt.subplot(1,3,3)
        sns.kdeplot(df['capital-gain'],bw=1.5)
        plt.title("KDE Plot - Capital-Gain")
        
        st.pyplot()

    if(st.checkbox('Capital-Loss')):
        # Capital-loss

        plt.figure(figsize=(15,5))
        
        plt.subplot(1,3,1)
        plt.boxplot(df['capital-loss'])
        plt.title("Box Plot - Capital-Loss")
        
        plt.subplot(1,3,2)
        plt.hist(df['capital-loss'],histtype='step')
        plt.title("Histograph - Capital-Loss")
        
        plt.subplot(1,3,3)
        sns.kdeplot(df['capital-loss'],bw=1.5)
        plt.title("KDE Plot - Capital-Loss")
        
        st.pyplot()
        
    if(st.checkbox('Hours-Per-Week')):
        # Hours Per Week

        plt.figure(figsize=(15,10))
        
        plt.subplot(1,3,1)
        plt.title("Box Plot - hours-per-week")
        plt.boxplot(df['hours-per-week'])
        
        plt.subplot(1,3,2)
        plt.title("Histographs - hours-per-week")
        plt.hist(df['hours-per-week'])
        
        plt.subplot(1,3,3)
        plt.title("PDF - hours-per-week")
        sns.distplot(df['hours-per-week'],hist=False, rug=False)
        st.pyplot()
        
def load_uni_cat(df):
    st.subheader('Univariate Analysis of Numerical Columns')
    if(st.checkbox('Workclass')):
        plt.figure(figsize=(15,10))
        
        plt.subplot(1,2,1)
        sns.countplot(data=df, x = 'workclass')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        df.workclass.value_counts().plot(kind='pie',
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
        
        st.pyplot()
        
    if(st.checkbox('Education')):
        plt.figure(figsize=(15,10))
        
        plt.subplot(1,2,1)
        sns.countplot(data=df, x = 'education')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        df.education.value_counts().plot(kind='pie',
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
        
        st.pyplot()
        
    if(st.checkbox('Marital-Status')):
        plt.figure(figsize=(15,10))
        
        plt.subplot(1,2,1)
        sns.countplot(data=df, x = 'marital-status')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        df['marital-status'].value_counts().plot(kind='pie',
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
        
        st.pyplot()
        
    if(st.checkbox('Occupation')):
        plt.figure(figsize=(15,10))
        
        plt.subplot(1,2,1)
        sns.countplot(data=df, x = 'occupation')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        df.occupation.value_counts().plot(kind='pie',
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
        
        st.pyplot()
        
    if(st.checkbox('Relationship')):
        plt.figure(figsize=(15,10))
        
        plt.subplot(1,2,1)
        sns.countplot(data=df, x = 'relationship')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        df.relationship.value_counts().plot(kind='pie',
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
        
        st.pyplot()
        
    if(st.checkbox('Race')):
        plt.figure(figsize=(15,10))
        
        plt.subplot(1,2,1)
        sns.countplot(data=df, x = 'race')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        df.race.value_counts().plot(kind='pie',
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
        
        st.pyplot()
        
    if(st.checkbox('Gender')):
        plt.figure(figsize=(15,10))
        
        plt.subplot(1,2,1)
        sns.countplot(data=df, x = 'gender')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        df.gender.value_counts().plot(kind='pie',
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
        
        st.pyplot()
        
    if(st.checkbox('Native-Country')):
        plt.figure(figsize=(15,10))
        
        plt.subplot(1,2,1)
        sns.countplot(data=df, x = 'native-country')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        df['native-country'].value_counts().plot(kind='pie',
                              wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                              textprops={'fontsize': 15})
        
        st.pyplot()
        
def load_univariate(df):
    st.header('Univariate Analysis')
    if(st.checkbox('For Numerical Columns')):
        load_uni_num(df)
    elif(st.checkbox('For Categorical Columns')):
        load_uni_cat(df)
        
def load_bi_num(df):
    if(st.checkbox('Pair Plot')):
        plt.figure(figsize=(15,10))
        sns.pairplot(df, vars=['age', 'educational-num', 'hours-per-week'],hue='income', corner=True, palette='husl')
        st.pyplot()
        
    if(st.checkbox('Scatter Plots')):
        plt.figure(figsize=(15,10))
        plt.subplot(2,2,1)
        plt.title("Scatter Plot - Age vs Capital-gain")
        sns.scatterplot(x = df['capital-gain'], y = df['age'], hue = df['income'], palette='Dark2')
        plt.subplot(2,2,2)
        plt.title("Scatter Plot - Age vs Capital-loss")
        sns.scatterplot(x = df['capital-loss'], y = df['age'], hue = df['income'], palette='Dark2')
        plt.subplot(2,2,3)
        plt.title("Scatter Plot - Hours-Per-Week vs Capital-gain")
        sns.scatterplot(x = df['capital-gain'], y = df['hours-per-week'], hue = df['income'], palette='Dark2')
        plt.subplot(2,2,4)
        plt.title("Scatter Plot - Hours-Per-Week vs Capital-gain")
        sns.scatterplot(x = df['capital-loss'], y = df['hours-per-week'], hue = df['income'], palette='Dark2')
        st.pyplot()
    if(st.checkbox('Join Plots')):
        plt.figure(figsize=(15,10))
        
        plt.subplot(1,2,1)
        plt.title('Age vs Hours-Per-Week')
        sns.jointplot(x='age', y='hours-per-week',kind= 'hex', data=df, color='k')
        st.pyplot()
        plt.subplot(1,2,2)
        plt.title('Capital-Gain vs Capital-loss')
        sns.jointplot(x='capital-gain', y='capital-loss',kind= 'hex', data=df, color='k')
        
        st.pyplot()
            
    
def load_bi_cat(df):
    plt.figure(figsize=(15,10))
    plt.subplot(1,3,1)
    sns.countplot(df['workclass'], hue=df['income'],palette="husl")
    plt.xticks(rotation=45)
    plt.subplot(1,3,2)
    sns.countplot(df['occupation'], hue=df['income'],palette="husl")
    plt.xticks(rotation=45)
    plt.subplot(1,3,3)
    sns.countplot(df['marital-status'], hue=df['income'],palette="husl")
    plt.xticks(rotation=45)
    st.pyplot()   
    
    
def load_num_cat(df):
    if(st.checkbox('Box-Plots')):
        plt.figure(figsize=(15,10))
        plt.subplot(2,2,1)
        plt.title("Age vs Income")
        sns.boxplot(data = df, x='income', y='age')
        
        plt.subplot(2,2,2)
        plt.title("Capital-Gain vs Income")
        sns.boxplot(data = df, x='income', y='capital-gain')
        
        plt.subplot(2,2,3)
        plt.title("Capital-Loss vs Income")
        sns.boxplot(data = df, x='income', y='capital-loss')
        
        plt.subplot(2,2,4)
        plt.title("Hours-Per-Week vs Income")
        sns.boxplot(data = df, x='income', y='hours-per-week')
        
        st.pyplot()
    
    if(st.checkbox('Bar Plots')):
        plt.figure(figsize=(15,10))
        plt.subplot(2,2,1)
        plt.title("Age vs Income")
        sns.barplot(data = df, x='income', y='age')
        
        plt.subplot(2,2,2)
        plt.title("Capital-Gain vs Income")
        sns.barplot(data = df, x='income', y='capital-gain')
        
        plt.subplot(2,2,3)
        plt.title("Capital-Loss vs Income")
        sns.barplot(data = df, x='income', y='capital-loss')
        
        plt.subplot(2,2,4)
        plt.title("Hours-Per-Week vs Income")
        sns.barplot(data = df, x='income', y='hours-per-week')
        st.pyplot()
        
    if(st.checkbox('Line Plots')):
        plt.figure(figsize=(15, 10))
        plt.subplot(2,1,1)
        plt.title('Educational-Num vs Age w.r.t education')
        sns.lineplot(x = 'age', y = 'educational-num', hue='education', data = df)
        
        plt.subplot(2,1,2)
        plt.title('Educational-Num vs Age w.r.t income')
        sns.lineplot(x = 'age', y = 'educational-num', hue='income', data = df)
        
        st.pyplot()
        
        
def load_bivariate(df):
    st.header('Bivariate Analysis')
    df=df.dropna()
    df = df.drop(columns = "fnlwgt")
    if(st.checkbox('Relationship between two Numerical Columns')):
        load_bi_num(df)
    elif(st.checkbox('Relationship between two Categorical Columns')):
        load_bi_cat(df)
    elif(st.checkbox('Relationship between categorical and numerical features')):
        load_num_cat(df)



def prdit(data):
    # sc = load(open('pickle/scaler.pkl', 'rb'))
    # modl = load(open("pickle/modl.pkl","rb"))
    st.write(data)
    
    
def load_predictor(df):
    st.subheader('Income Prediction')
    st.info("educational-num")
    edu_num = st.slider("",0,10,1)
    st.info("capital-gain")
    cap_gain = st.number_input("")
    st.info("education")
    edu = st.radio("",("education_5th-6th","education_Preschool","Other"))
    if(edu == "Other"):
        edu_5_6 = 0
        edu_pre = 0
    elif(edu == "education_5th-6th"):
        edu_5_6 = 1
        edu_pre = 0
    elif (edu == "education_Preschool"):
        edu_5_6 = 0
        edu_pre = 1

    st.info("marital-status")
    mar_status = st.radio("",("Married-civ-spouse","Married-AF-spouse","Other"))
    if(mar_status == "Other"):
        mar_civ_sp = 0
        mar_af_sp = 0
    elif(mar_status == "Married-civ-spouse"):
        mar_civ_sp = 1
        mar_af_sp = 0
    elif(mar_status == "Married-AF-spouse"):
        mar_civ_sp = 0
        mar_af_sp = 1

    st.info("occupation")
    occ = st.radio("",("occupation_Farming-fishing","occupation_Handlers-cleaners",
                        "occupation_Other-service","occupation_Priv-house-serv","Other"))
    if(occ == "occupation_Farming-fishing"):
        occ_far_fish = 1
        occ_han_clean = 0
        occ_oth_sr = 0
        occ_pr_hous_srv = 0
    elif (occ == "occupation_Handlers-cleaners"):
        occ_far_fish = 0
        occ_han_clean = 1
        occ_oth_sr = 0
        occ_pr_hous_srv = 0
    elif (occ == "occupation_Other-service"):
        occ_far_fish = 0
        occ_han_clean = 0
        occ_oth_sr = 1
        occ_pr_hous_srv = 0
    elif (occ == "occupation_Priv-house-serv"):
        occ_far_fish = 0
        occ_han_clean = 0
        occ_oth_sr = 0
        occ_pr_hous_srv = 1
    elif (occ == "Other"):
        occ_far_fish = 0
        occ_han_clean = 0
        occ_oth_sr = 0
        occ_pr_hous_srv = 0

    st.info("Relationship")
    rel = st.radio("",("relationship_Other-relative","relationship_Own-child","Other"))
    if(rel == "relationship_Other-relative"):
        rel_oth_rel = 1
        rel_own_chl = 0
    elif (rel == "relationship_Other-relative"):
        rel_oth_rel = 0
        rel_own_chl = 1
    elif(rel == "Other"):
        rel_oth_rel = 0
        rel_own_chl = 0

    st.info("Native")
    nat = st.radio("",("native-country_Columbia","native-country_Dominican-Republic",
                            "native-country_Guatemala","native-country_Laos",
                            "native-country_Mexico","native-country_Nicaragua",
                            "native-country_South","native-country_Vietnam","Other"))
    if(nat == "native-country_Columbia"):
        nat_con_col = 1
        nat_con_dom = 0
        nat_con_gat = 0
        nat_con_laos = 0
        nat_con_m = 0
        nat_col_nicar = 0
        nat_col_south = 0
        nat_col_vitn = 0
    elif(nat == "native-country_Dominican-Republic"):
        nat_con_col = 0
        nat_con_dom = 1
        nat_con_gat = 0
        nat_con_laos = 0
        nat_con_m = 0
        nat_col_nicar = 0
        nat_col_south = 0
        nat_col_vitn = 0
    elif (nat == "native-country_Guatemala"):
        nat_con_col = 0
        nat_con_dom = 0
        nat_con_gat = 1
        nat_con_laos = 0
        nat_con_m = 0
        nat_col_nicar = 0
        nat_col_south = 0
        nat_col_vitn = 0
    elif (nat == "native-country_Laos"):
        nat_con_col = 0
        nat_con_dom = 0
        nat_con_gat = 0
        nat_con_laos = 1
        nat_con_m = 0
        nat_col_nicar = 0
        nat_col_south = 0
        nat_col_vitn = 0
    elif (nat == "native-country_Mexico"):
        nat_con_col = 0
        nat_con_dom = 0
        nat_con_gat = 0
        nat_con_laos = 0
        nat_con_m = 1
        nat_col_nicar = 0
        nat_col_south = 0
        nat_col_vitn = 0
    elif (nat == "native-country_Nicaragua"):
        nat_con_col = 0
        nat_con_dom = 0
        nat_con_gat = 0
        nat_con_laos = 0
        nat_con_m = 0
        nat_col_nicar = 1
        nat_col_south = 0
        nat_col_vitn = 0
    elif (nat == "native-country_South"):
        nat_con_col = 0
        nat_con_dom = 0
        nat_con_gat = 0
        nat_con_laos = 0
        nat_con_m = 0
        nat_col_nicar = 0
        nat_col_south = 1
        nat_col_vitn = 0
    elif (nat == "native-country_South"):
        nat_con_col = 0
        nat_con_dom = 0
        nat_con_gat = 0
        nat_con_laos = 0
        nat_con_m = 0
        nat_col_nicar = 0
        nat_col_south = 0
        nat_col_vitn = 1
    elif (nat == "Other"):
        nat_con_col = 0
        nat_con_dom = 0
        nat_con_gat = 0
        nat_con_laos = 0
        nat_con_m = 0
        nat_col_nicar = 0
        nat_col_south = 0
        nat_col_vitn = 0
    data = [edu_num,cap_gain,edu_5_6,edu_pre,mar_af_sp,mar_civ_sp,occ_far_fish,occ_han_clean,
            occ_oth_sr,occ_pr_hous_srv,rel_oth_rel,rel_own_chl,nat_con_col,nat_con_dom,nat_con_gat,
            nat_con_laos,nat_con_m,nat_col_nicar,nat_col_south,nat_col_vitn]
    pr = prdit(data)
    st.write(pr)
    
        
def load_sidebar(df):
    st.sidebar.subheader("Adult Income Prediction")
    st.sidebar.success(' To download the dataset :https://drive.google.com/file/d/17yZ1NSSsRrDF7qfGOJyGHRDDiBce7EQN/view')
    st.sidebar.info('Detailed Description about the dataset : http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html')
    st.sidebar.success('For EDA, Model Building and Web App code : https://github.com/pooja1207/CIPHERSCHOOLS__ML1 ')
    st.sidebar.warning('Made by Pooja Bhather :heart:')

def main():
    df = load_data(dataset_loc)
    
    # Title/ text
    st.title('Adult Income Prediction')
    st.image("data/img_1.png", use_column_width = True)
    st.text('Predict the annual income of a individual whether it is less than or greater than 50K')
    
    load_sidebar(df)
    
    
    add_selectbox = st.selectbox('What would you like to see?',('Choose an option', 'Introduction', 'Data Description', 'Missing Value', 'Univariate Analysis', 'Bivariate Analysis', 'Income Prediction'))
    if(add_selectbox == 'Introduction'):
        load_intro(df)
    elif(add_selectbox == 'Data Description'):
        load_description(df)
    elif(add_selectbox == 'Missing Value'):
        load_missing_values(df)
    elif(add_selectbox == 'Univariate Analysis'):
        load_univariate(df)
    elif(add_selectbox == 'Bivariate Analysis'):
        load_bivariate(df)
    elif(add_selectbox == 'Income Prediction'):
        load_predictor(df)

if(__name__ == '__main__'):
    main()