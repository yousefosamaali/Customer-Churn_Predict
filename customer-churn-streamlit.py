import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import pickle
import mlflow
import mlflow.keras



    # Preprocessing function
def preprocess(df, numerical, categorical):
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()

    numerical_pipeline = Pipeline(steps=[('scaler', scaler)])

    categorical_pipeline = Pipeline(steps=[('encoder', ohe)])

    transformer = ColumnTransformer(
        transformers=[
            ('numerical', numerical_pipeline, numerical),
            ('categorical', categorical_pipeline, categorical)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    X_processed = transformer.fit_transform(df)
    # Explicitly convert the transformed data to float32
    X_processed = X_processed.astype(np.float32)
    columns = transformer.get_feature_names_out()
    return X_processed, columns, transformer



# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
    df =df.drop(['customerID'], axis=1)
    return df

df = load_data()
categorical = list(df.select_dtypes(include=['object']).columns)
numerical = list(df.select_dtypes(include=['number']).columns)[:-1]
df_c = df.drop(['Churn'],axis=1)
X_transformed, feature_columns, trans = preprocess(df_c, numerical, categorical)


# Sidebar
st.sidebar.title('Churn Data Exploration and Model Training')
options = st.sidebar.selectbox('Choose an Option', ['Data Visualization', 'Train Model', 'Predict'])

# Data visualization
if options == 'Data Visualization':
    st.title('Data Visualization')
    if st.checkbox('Show Raw Data'):
        st.dataframe(df.head())

    st.subheader('Churn Count Plot')
    fig, ax = plt.subplots()
    sns.countplot(x=df['Churn'], ax=ax)
    st.pyplot(fig)
############################################
    st.subheader('Churn by Contract Type')
    fig, ax = plt.subplots()
    sns.countplot(x='Contract', hue='Churn', data=df, ax=ax)
    st.pyplot(fig)
##########################################
    st.subheader('Churn by Selected Feature')
    y_axis = st.selectbox('Select a variable for the Y-axis:', ['MonthlyCharges', 'TotalCharges', 'tenure'])
    fig, ax = plt.subplots()
    sns.boxplot(x='Churn', y=y_axis, data=df, ax=ax)
    st.pyplot(fig)
    ################################
    st.subheader('Numric Columns Distribution by Churn')
    x_axis = st.selectbox('Select a variable for the X-axis:', ['MonthlyCharges', 'TotalCharges', 'tenure'])
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=x_axis, hue='Churn', multiple='stack', bins=30)
    st.pyplot(fig)
#########################################
    st.subheader('PaymentMethod relationship with Churn')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Bar plot of PaymentMethod vs. Churn
    sns.countplot(x='PaymentMethod', hue='Churn', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Payment Method vs. Churn (Count Plot)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Pie chart of PaymentMethod distribution
    df['PaymentMethod'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[0, 1])
    axes[0, 1].set_title('Payment Method Distribution')
    axes[0, 1].set_ylabel('')

    # Boxplot showing MonthlyCharges by PaymentMethod
    sns.boxplot(x='PaymentMethod', y='MonthlyCharges', hue='Churn', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Monthly Charges by Payment Method and Churn')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Point plot showing relationship between PaymentMethod and Churn Rate
    df_grouped = df.groupby('PaymentMethod')['Churn'].mean().reset_index()
    sns.pointplot(x='PaymentMethod', y='Churn', data=df_grouped, ax=axes[1, 1])
    axes[1, 1].set_title('Churn Rate by Payment Method')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    ###############################################
    global_mean = df['Churn'].mean()


    st.subheader("Churn Rate by Categorical Features")

    feature = st.selectbox('Select a variable for the X-axis:', categorical)
    df_group = df.groupby(by=feature).Churn.agg(['mean']).reset_index()


    fig, ax = plt.subplots(figsize=(5, 3))


    graph = sns.barplot(x=feature, y='mean', data=df_group, palette='Greens', ax=ax)


    ax.axhline(global_mean, linewidth=3, color='b')
    

    ax.text(0, global_mean - 0.03, "global_mean", color='black', weight='semibold')


    ax.set_title(f'Churn Rate by {feature}')
    ax.set_ylabel('Mean Churn Rate')
    ax.set_xlabel(feature)

    st.pyplot(fig)
    ############################################
    correlation_matrix = df[numerical].corr()
    st.subheader('Correlation Matrix of Numeric Features and Churn')
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix of Numeric Features and Churn')
    st.pyplot(fig)

# Model training
elif options == 'Train Model':
    st.title('Train the Model')
    X_transformed.shape
    df_x = df.drop(['Churn'],axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(df_x, y, test_size=0.2, random_state=42, shuffle=True)
    X_train, _, tf = preprocess(X_train, numerical, categorical)
    X_test = tf.transform(X_test)

    # Model
    model = Sequential()
    model.add(Dense(16, input_dim=X_transformed.shape[1], activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))  

    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))  
    

    model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))  


    model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))  

    
    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if st.button('Train Model'):
        
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), 
                            verbose=1)
        st.success('Model trained!')

        # Displaying accuracy
        _, accuracy = model.evaluate(X_test, y_test)
        st.write(f'Test Accuracy: {accuracy * 100:.2f}%')

        # Save the model
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)

        # Plot the model accuracy and loss
        st.subheader('Training History')
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
        ax[0].legend()
        ax[0].set_title('Accuracy')

        ax[1].plot(history.history['loss'], label='Train Loss', color='green')
        ax[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
        ax[1].legend()
        ax[1].set_title('Loss')
        st.pyplot(fig)
        
        y_pred = model.predict(X_test)
        y_pred_classes = np.round(y_pred).astype(int)
        report = classification_report(y_test, y_pred_classes, target_names=['No', 'Yes'], output_dict=True)

        # Streamlit can display dataframes, so we convert the classification report to a dataframe for better display
        report_df = pd.DataFrame(report).transpose()

        # Display the classification report in Streamlit
        st.subheader('Classification Report')
        st.dataframe(report_df)
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_test, y_pred_classes)
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
    if st.button('Train by K-Fold Cross-Validation'):
        

        # Define K-Fold Cross Validator
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y = df['Churn']
        # Store accuracies for each fold
        accuracies = []
        classification_reports = []

        # K-Fold Cross Validation
        for train_index, test_index in kf.split(X_transformed):
            X_train, X_test = X_transformed[train_index], X_transformed[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Build Model
            model = Sequential()
            model.add(Dense(16, input_dim=X_transformed.shape[1], activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))  
            
            model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))  
            
            model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))  
            
            model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))  
            
            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Train Model
            history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

            # Evaluate Model
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            accuracies.append(accuracy)

            st.subheader('Training History')
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            ax[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
            ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
            ax[0].legend()
            ax[0].set_title('Accuracy')

            ax[1].plot(history.history['loss'], label='Train Loss', color='green')
            ax[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
            ax[1].legend()
            ax[1].set_title('Loss')
            st.pyplot(fig)
            
            # Generate classification report
            y_pred = model.predict(X_test)
            y_pred_classes = np.round(y_pred).astype(int)
            report = classification_report(y_test, y_pred_classes, target_names=['No', 'Yes'], output_dict=True)
            classification_reports.append(report)

        # Display Average Accuracy
        st.write(f'Average Accuracy across all folds: {np.mean(accuracies) * 100:.2f}%')

        # Optionally, display the classification reports for each fold
        for i, report in enumerate(classification_reports):
            st.subheader(f'Classification Report for Fold {i + 1}')
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)



if options == 'Predict':
    st.title('Predict Churn for a New Customer')

    # Create form for user input
    with st.form("customer_form"):
        gender = st.selectbox('Gender', options=['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', options=['Yes', 'No'])
        partner = st.selectbox('Partner', options=['Yes', 'No'])
        dependents = st.selectbox('Dependents', options=['Yes', 'No'])
        tenure = st.number_input('Tenure (Months)', min_value=0, max_value=72, step=1)
        phone_service = st.selectbox('Phone Service', options=['Yes', 'No'])
        multiple_lines = st.selectbox('Multiple Lines', options=['Yes', 'No', 'No phone service'])
        internet_service = st.selectbox('Internet Service', options=['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', options=['Yes', 'No', 'No internet service'])
        online_backup = st.selectbox('Online Backup', options=['Yes', 'No', 'No internet service'])
        device_protection = st.selectbox('Device Protection', options=['Yes', 'No', 'No internet service'])
        tech_support = st.selectbox('Tech Support', options=['Yes', 'No', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', options=['Yes', 'No', 'No internet service'])
        streaming_movies = st.selectbox('Streaming Movies', options=['Yes', 'No', 'No internet service'])
        contract = st.selectbox('Contract', options=['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', options=['Yes', 'No'])
        payment_method = st.selectbox('Payment Method', options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly_charges = st.number_input('Monthly Charges', min_value=0, step=1)
        total_charges = st.number_input('Total Charges', min_value=0, step=1)
        #Churn = st.selectbox('Churn', options=['Yes', 'No'])
        submit_button = st.form_submit_button(label='Predict Churn')

    # When the user submits the form
    a = np.random.randint(0, 2)
    if submit_button:
        # Create a DataFrame for the new user input
        user_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            #'Churn' : [1]
        })

        # Preprocess user input with the same transformer used for training data
        user_data['TotalCharges'] = pd.to_numeric(user_data['TotalCharges'], errors='coerce')
        user_data['TotalCharges'].fillna(user_data['TotalCharges'].median(), inplace=True)
        #user_data['Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)
        user_data['SeniorCitizen'] = user_data['SeniorCitizen'].astype('object')
        
        user_input_transformed = trans.transform(user_data)
        print(user_input_transformed.shape)
        #user_input_transformed = user_input_transformed[:, :-1]
        print(pd.DataFrame(user_input_transformed))
        # Load trained model
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        # Predict the probability of churn
        prediction = model.predict(user_input_transformed)
        #prediction = (prediction > 0.5).astype(int)  # Threshold for binary classification
        print(prediction)
        
        st.write(f"The customer is {round(prediction[0][0], 2)*100} is going to churn.")

            
