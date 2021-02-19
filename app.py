import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler,scale, LabelEncoder
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA 

from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

################################################
#
# Homepage
#
################################################

page = st.sidebar.selectbox("Choose a page", ["Exploration", "Classification"])


################################################
#
# Load Dataset and Data Preparation for Modeling
#
################################################
# @st.cache 
def load_data():
    data = pd.read_csv('weatherAUS.csv', header=0)
    data = data.dropna().reset_index(drop=True)
    return data



def clean_data(df):
    df.drop(['Date','Location'],axis=1, inplace=True)

    # Clean catetorical variables' levels
    cat_cols = ['WindDir9am', 'WindDir3pm', 'WindGustDir']
    for col in cat_cols:
        df.loc[(df[col] == 'SSW')|(df[col] == 'SSE'), col] = 'S'
        df.loc[(df[col] == 'NNE')|(df[col] == 'NNW'), col] = 'N'
        df.loc[(df[col] == 'WSW')|(df[col] == 'WNW'), col] = 'W'
        df.loc[(df[col] == 'ENE')|(df[col] == 'ESE'), col] = 'E'

    # transform: Label Encoding
    label_encoder = LabelEncoder()
    # Check the labels
    label_encoder.fit(df['RainTomorrow'])
    # st.write("Level of classes: ",label_encoder.classes_ )
    df['RainTomorrow'] = label_encoder.transform(df.RainTomorrow)
    # Dummy Coding
    cat_vars = ['WindDir9am', 'WindDir3pm', 'WindGustDir', 'RainToday']

    for var in cat_vars:
        cat_list = pd.get_dummies(df[var], prefix=var)
        df = df.join(cat_list)
    df_final = df.drop(cat_vars, axis=1)
    return df_final



def transform_data(df_final):
    transformed = dict()
    X, y = df_final.values[:, :16], df_final.values[:, 16]
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # Scale
    X_scaler = StandardScaler().fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    transformed["X"] = X
    transformed["y"] = y
    transformed["X_train_scaled"] = X_train_scaled
    transformed["X_test_scaled"] = X_test_scaled
    transformed["y_train"] = y_train
    transformed["y_test"] = y_test
    return transformed


################################################
#
# Build Classifiers
#
################################################
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 42)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.5, 100.0)
        gamma = st.sidebar.slider("gamma", 0.0001, 1.0)
        params["C"] = C
        params["gamma"] = gamma
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider('max_depth', 2, 42)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params




def get_classifier(clf_name, params):
    clf = None
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(solver='liblinear')
    elif clf_name == "SVM":
        clf = SVC(random_state=42, C=params["C"],gamma=params["gamma"])
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == "Decision Tree Classifier":
        clf = DecisionTreeClassifier()
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                        max_depth=params["max_depth"],
                                        random_state=42)
    
    return clf



def get_accuracy(clf, transformed):
    clf.fit(transformed["X_train_scaled"], transformed["y_train"])
    y_pred = clf.predict(transformed["X_test_scaled"])
    acc = accuracy_score(transformed["y_test"], y_pred)
    return acc


################################################
#
# PLOT part1
#
################################################
def visualize_y(df):
    st.write("Average by two levels: (0 - Not Rain, 1 - Rain)")
    st.write(df.groupby('RainTomorrow').mean())   

    count_no = len(df[df.RainTomorrow == 0])
    count_yes = len(df[df.RainTomorrow == 1])
    total = len(df)

    st.write(f"There are {len(df[df.RainTomorrow == 1])} days that will rain the next day. ", 
      f"There are {len(df[df.RainTomorrow == 0])} days that would not rain the next day.")
    st.write("Percentage of raining is about {0:.2%}; ".format(count_yes/total),
        " percentage of not raining is about {0:.2%}.".format(count_no/total))
    col_a, col_b = st.beta_columns([2,1])
    fig = plt.figure(figsize=(8,6))
    sns.countplot(x = 'RainTomorrow', data=df, palette='Spectral')
    plt.show()
    col_a.pyplot(fig)


def get_hist(df):
    st.markdown("Histogram")
    col1, col2 = st.beta_columns(2)
    with col1: 
        fig1 = plt.figure()
        df.Rainfall.hist(bins=50, color="olive")
        plt.title('Histogram of Rainfall')
        plt.xlim(-10,90)
        plt.xlabel('Rainfall(millimetres)')
        plt.ylabel('Frequency')
        st.pyplot(fig1)
    with col2: 
        fig2 = plt.figure()
        df.Evaporation.hist(bins=50, color="olive")
        plt.title('Histogram of Evaporation')
        plt.xlabel('Evaporation(millimetres)')
        plt.ylabel('Frequency')
        st.pyplot(fig2)
    col3, col4 = st.beta_columns(2)
    with col3: 
        fig3 = plt.figure()
        df.Sunshine.hist(bins=30, color="olive")
        plt.title('Histogram of Sunshine')
        plt.xlabel('Sunshine (hours)')
        plt.ylabel('Frequency')
        st.pyplot(fig3)
    with col4:
        fig4 = plt.figure()
        df.Cloud9am.hist(bins=10, color="olive")
        plt.title('Fraction of sky obscured by cloud at 9 am')
        plt.xlabel('Cloud9am (eighths)')
        plt.ylabel('Frequency')
        st.pyplot(fig4)

    st.markdown("Heatmap")
    fig = plt.figure()
    X = df.iloc[:, 0:26]
    y = df.iloc[:,-1]
    corrmat = df.corr()

    top_corr_features = corrmat.index
    fig = plt.figure(figsize=(20,20))
    #plot heat map
    sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn", linewidths=.5)
    st.pyplot(fig)
################################################
#
# PLOT part2
#
################################################
def pca(transformed):
    pca = PCA(2)
    X_projected = pca.fit_transform(transformed["X"])

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=transformed["y"], alpha = 0.4, cmap="crest")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    st.pyplot(fig)


################################################
#
# Pages
#
################################################
def main():
    df = load_data()
    df_final = clean_data(df)
    if page == "Exploration":
        st.title("Rain in Australia Prediction")
        st.markdown("""
        This project performs simple visualization and classification of the Rain in Australia data.
        * **Python libraries:** pandas, numpy, matplotlib, seaborn, streamlit, scikit-learn. 
        * **Data source:** [rain-in-australia-kaggle.com](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
        """)
        st.subheader("1. Display DataFrame")
        st.write(df_final.head(20))

        st.subheader("2. Outcome Variable")
        visualize_y(df_final)

        st.subheader("3. Visualization")
        get_hist(df)


    elif page == "Classification":
        st.title("Data Classification")
        st.write("""
        ### Explore different classifier
        *Which has the highest accuracy?*
        """)
        classifier_name = st.selectbox("Select Classifier",("Logistic Regression", "SVM","KNN", "Decision Tree Classifier","Random Forest"))
       
        df = load_data()
        df_final = clean_data(df)
        st.write("shape of dataset", df_final.shape)

        transformed = transform_data(df_final)
        params = add_parameter_ui(classifier_name)
        clf = get_classifier(classifier_name, params)
        acc = get_accuracy(clf,transformed)

        st.write(f"classifier = {classifier_name}")
        st.write(f'Accuracy =', acc)

        pca(transformed)

if __name__ == "__main__":
    main()