import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yellowbrick
from yellowbrick.target import ClassBalance
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import lightgbm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

sns.set(style="white", palette="muted", color_codes=True)
pd.set_option("display.width", 1000)

bs_df = pd.read_csv("C:/Users/emirh/Downloads/full_data.csv")
#### Data Analysis ####
class DataOperation:
    def __init__(self, df):
        self.df = df

    def get_columns(self):
        self.df.columns = self.df.columns.str.lower()
        discrete_cols = [
            var
            for var in self.df.columns
            if self.df[var].dtypes != "O" and self.df[var].nunique() < 10
        ]
        continuous_cols = [
            var
            for var in self.df.columns
            if self.df[var].dtypes != "O"
            and var != "stroke"
            and var not in discrete_cols
        ]
        cat_cols = [var for var in self.df.columns if self.df[var].dtypes == "O"]
        numerical_cols = [var for var in self.df.columns if self.df[var].dtypes != "O"]
        mixed_cols = discrete_cols + cat_cols
        return discrete_cols, continuous_cols, cat_cols, numerical_cols, mixed_cols

    def data_preview(self):

        print(self.df.head())
        print("*" * 40)
        print(f"Shape of data: {self.df.shape}")
        print("*" * 40)
        print(self.df.info())
        print("*" * 40)
        print(self.df.describe().T)

    def data_summary(self):
        (
            discrete_cols,
            continuous_cols,
            cat_cols,
            numerical_cols,
            mixed_cols,
        ) = self.get_columns()

        cardinality = self.df[cat_cols].nunique()

        print(f"Total Discrete Variables: {len(discrete_cols)} --> {discrete_cols}")
        print(
            f"Total Continuous Variables: {len(continuous_cols)} --> {continuous_cols}"
        )
        print(f"Total Categorical Variables: {len(cat_cols)} --> {cat_cols}")
        print("*" * 40)
        print(f"Cardinality: \n{cardinality}")
        for col in self.df[mixed_cols]:
            print("*" * 40)
            print(
                f"Counts of unique values for variables:\n\033[1m {col} \033[0m \n{self.df[col].value_counts()}"
            )

    def data_corr(self):

        corr = self.df.corr()
        sns.heatmap(corr, annot=True, cmap="RdBu", vmin=-1, vmax=1, fmt=".2f")

    def missing_values_analysis(self):

        if self.df.isnull().sum().any():
            missing_percent = self.df.isnull().sum() / len(self.df) * 100
            missing_total = self.df.isnull().sum()
            missing_table = pd.concat([missing_total, missing_percent], axis=1)
            missing_table.rename(
                columns={0: "Missing Value", 1: "% of Total Value"}, inplace=True
            )
            missing_table.sort_values(by="% of Total Value", ascending=False).round(1)
        else:
            print("There is no missing value.")

    def numerical_viz(self):
        (
            discrete_cols,
            continuous_cols,
            cat_cols,
            numerical_cols,
            mixed_cols,
        ) = self.get_columns()

        for col in self.df[continuous_cols]:

            fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
            sns.set_color_codes(palette="pastel")
            sns.histplot(x=self.df[col], ax=ax[0], kde=True).set(
                title=f"Distribution of {col}"
            )
            sns.boxplot(x=self.df[col], ax=ax[1]).set(title=f"Box-plot of {col}")
            sns.despine(left=True)
            plt.show()

    def categorical_viz(self):
        (
            discrete_cols,
            continuous_cols,
            cat_cols,
            numerical_cols,
            mixed_cols,
        ) = self.get_columns()
        c = 1
        fig = plt.figure(figsize=(15, 10))
        for col in self.df[mixed_cols]:

            plt.subplot(2, 4, c)
            sns.countplot(x=self.df[col]).set(title=f"Plot of {col}")
            plt.xticks(rotation=30)
            sns.despine()
            c = c + 1
        plt.tight_layout()
        plt.show()

    def skewness_kurtosis(self):
        (
            discrete_cols,
            continuous_cols,
            cat_cols,
            numerical_cols,
            mixed_cols,
        ) = self.get_columns()
        skew = {}
        kurt = {}
        for col in self.df[continuous_cols]:
            skewness = self.df[col].skew(axis=0).round(2)
            kurtosis = self.df[col].kurtosis(axis=0).round(2)
            skew[col] = skewness
            kurt[col] = kurtosis
        skw_kurt = pd.DataFrame({"skew": skew, "kurtosis": kurt}).T
        print(skw_kurt)
        for col in skw_kurt.columns:
            if ((skw_kurt[col].iloc[0] > 1.5) | (skw_kurt[col].iloc[0] < -1.5)) and (
                (skw_kurt[col].iloc[1] > 1.5) | (skw_kurt[col].iloc[1] < -1.5)
            ):
                print("*" * 30)
                print(
                    f"\033[1mSkewness and kurtosis were detected.\033[0m\nVariable: {col}\n\033[1mWe'll apply Box-Cox transformation.\033[0m"
                )
                self.df[col + "_boxcox"], df_lambda = stats.boxcox(self.df[col])
                self.df.drop([col], axis=1, inplace=True)
                continuous_cols.remove(col)
                continuous_cols += [col + "_boxcox"]
                self.numerical_viz()

    def cat_encoder(self):

        self.df["gender"] = [1 if i == "Male" else 0 for i in self.df["gender"]]
        self.df["ever_married"] = [
            1 if i == "Yes" else 0 for i in self.df["ever_married"]
        ]
        self.df["residence_type"] = [
            1 if i == "Urban" else 0 for i in self.df["residence_type"]
        ]
        if self.df["hypertension"].dtype == "O":
            self.df["hypertension"] = [
                1 if i == "Yes" else 0 for i in self.df["hypertension"]
            ]
        elif self.df["heart_disease"].dtype == "O":
            self.df["heart_disease"] = [
                1 if i == "Yes" else 0 for i in self.df["heart_disease"]
            ]
        else:
            pass

        self.df = pd.get_dummies(
            self.df, columns=["work_type", "smoking_status"], drop_first=True
        )
        return self.df

    def get_train_test_data(self, imbalance=True):
        X = self.df.drop(["stroke"], axis=1)
        y = self.df["stroke"]
        classes = y.unique()
        X_cols = X.columns
        visualizer = ClassBalance(labels=classes)
        print(y.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        if imbalance:

            smt = SMOTE(random_state=42)
            X_smt, y_smt = smt.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_smt, y_smt, test_size=0.2, random_state=42
            )
            plt.subplot(1, 2, 1)
            visualizer.fit(y)
            plt.title("Before Oversampling Target")

            plt.subplot(1, 2, 2)
            ClassBalance(labels=classes).fit(y_smt)
            plt.title("After Oversampling Target")

            plt.show()

            return X_train, X_test, y_train, y_test, X_cols
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            return X_train, X_test, y_train, y_test, X_cols


# bs_op = DataOperation()
# bs_op.data_preview()
# bs_op.data_summary()
# bs_op.data_corr()
# bs_op.missing_values_analysis()
# bs_op.numerical_viz()
# bs_op.categorical_viz()
# bs_op.skewness_kurtosis()
# bs_op.cat_encoder()
# train_test=bs_op.get_train_test_data()

############ MODEL BUILDING ############
class modelBuilding:
    def __init__(self):

        pass

    def gb_models(self):

        models = {
            "Xgboost": XGBClassifier(random_state=42),
            "LightGBM": LGBMClassifier(random_state=42),
            "CatBoost": CatBoostClassifier(random_state=42, verbose=False),
        }
        params = [
            {
                "n_estimators": [20, 50, 100, 200, 500],
                "max_depth": [2, 4, 6, 8],
                "learning_rate": [0.001, 0.01, 0.02, 0.04, 0.1, 0.2, 0.3, 0.5],
            }
        ]

        return models, params

    def fine_tuning(self):
        global train_test
        X_train, X_test, y_train, y_test, X_cols = train_test
        models, params = self.gb_models()
        best_param = {}

        for name, model in models.items():
            grid_search = GridSearchCV(
                model,
                params,
                cv=5,
                scoring="accuracy",
                return_train_score=True,
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            print(
                f"{name}: \nThe best score: {grid_search.best_score_} \nThe best param: {grid_search.best_params_}"
            )
            best_param[name] = grid_search.best_params_
        return best_param

    def model_evaluation(self, model_name, n_jobs=-1):
        global best_param
        X_train, X_test, y_train, y_test, X_cols = train_test

        if model_name == "Xgboost":
            print("-----------\033[1mXGBOOST\033[0m-----------")
            clf = XGBClassifier(
                n_estimators=best_param["Xgboost"]["n_estimators"],
                max_depth=best_param["Xgboost"]["max_depth"],
                learning_rate=best_param["Xgboost"]["learning_rate"],
                n_jobs=n_jobs,
            )
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            joblib.dump(clf, "xgb_bs_model.pkl")
            print(f"Train Data Accuracy Score: {accuracy_score(y_train,y_pred_train)}")
            print(f"Test Data Accuracy Score: {accuracy_score(y_test,y_pred_test)}")
            print("-" * 30)
            print(
                f"Classification Report: \n\n{classification_report(y_test,y_pred_test)}"
            )
            xgb_sorted_idx = clf.feature_importances_.argsort()
            plt.barh(X_cols[xgb_sorted_idx], clf.feature_importances_[xgb_sorted_idx])
            plt.xlabel("Feature Importance Score")
            plt.ylabel("Features")
            plt.title("Xgboost Feature Importance")
        elif model_name == "LightGBM":
            print("-----------\033[1mLightGBM\033[0m-----------")
            clf = LGBMClassifier(
                n_estimators=best_param["LightGBM"]["n_estimators"],
                max_depth=best_param["LightGBM"]["max_depth"],
                learning_rate=best_param["LightGBM"]["learning_rate"],
                n_jobs=n_jobs,
                verbose_eval=False,
            )
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            joblib.dump(clf, "lgb_bs_model.pkl")
            print("-----------\033[1mChecking Overfitting\033[0m-----------")
            print(f"Train Data Accuracy Score: {accuracy_score(y_train,y_pred_train)}")
            print(f"Test Data Accuracy Score: {accuracy_score(y_test,y_pred_test)}")
            print("-" * 30)
            print(
                f"Classification Report: \n\n{classification_report(y_test,y_pred_test)}"
            )
            lightgbm.plot_importance(clf)
            plt.xlabel("Feature Importance")
            plt.ylabel("Features")
            plt.title("LightGBM Feature Importance")
        elif model_name == "CatBoost":
            print("-----------\033[1mCatBoost\033[0m-----------")
            clf = LGBMClassifier(
                n_estimators=best_param["CatBoost"]["n_estimators"],
                max_depth=best_param["CatBoost"]["max_depth"],
                learning_rate=best_param["CatBoost"]["learning_rate"],
                n_jobs=n_jobs,
            )
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            joblib.dump(clf, "catb_bs_model.pkl")
            print("-----------\033[1mChecking Overfitting\033[0m-----------")
            print(f"Train Data Accuracy Score: {accuracy_score(y_train,y_pred_train)}")
            print(f"Test Data Accuracy Score: {accuracy_score(y_test,y_pred_test)}")
            print("-" * 30)
            print(
                f"Classification Report: \n\n{classification_report(y_test,y_pred_test)}"
            )
            cat_sorted_idx = clf.feature_importances_.argsort()
            plt.barh(X_cols[cat_sorted_idx], clf.feature_importances_[cat_sorted_idx])
            plt.xlabel("Feature Importance")
            plt.ylabel("Features")
            plt.title("Catboost Feature Importance")
        else:
            raise ValueError(
                f"Invalid Model \nPlease enter only one of the following model names: \nXgboost \nLightGBM \nCatboost"
            )



# bs_mb = modelBuilding()
# bs_mb.gb_models()
# bs_mb.fine_tuning()
# bs_mb.model_evaluation("LightGBM")
