# Title: DC Residential Property Valuation Analysis
# Author: Alexander Zakrzeski
# Date: February 13, 2026

# Part 1: Setup and Configuration

# Load to import, clean, and wrangle data
import geopandas as gpd
import polars as pl

# Load to generate correlation ratios
from dython.nominal import correlation_ratio

# Load to train, test, and evaluate machine learning models
from lightgbm import LGBMRegressor
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, r2_score, root_mean_squared_error
    )
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sktime.performance_metrics.forecasting import (
    median_absolute_percentage_error
    )
from xgboost import XGBRegressor

# Part 2: Function Definitions

def assign_high_school_zone(df, years, filename):
    """
    Return a DataFrame with a new column indicating the high-school attendance
    zone of each house for the specified years, produced by a spatial left join.
    """
    df = df.filter(pl.col("sale_year").is_in(years)).to_pandas()
    
    df = (
        gpd.sjoin(gpd.GeoDataFrame(
                      df, 
                      geometry = gpd.points_from_xy(df["longitude"], 
                                                    df["latitude"]),
                      crs = "EPSG:4326"
                      ),
                  gpd.read_file(filename)
                     .rename(columns = {"NAME": "high_school"})
                     [["geometry", "high_school"]],
                  how = "left", predicate = "within")
           .drop(columns = ["geometry", "index_right"])
        )
           
    df = (
        pl.from_pandas(df) 
          .with_columns(
              pl.when(pl.col("high_school") == "Wilson")
                .then(pl.lit("Jackson-Reed")) 
                .otherwise("high_school") 
                .alias("high_school")    
              )    
        )             
        
    return df

def assign_census_tract(df):
    """
    Return a DataFrame with a new column indicating the Census tract of each 
    house, produced by a spatial left join.
    """
    df = df.to_pandas()
    
    df = (
        gpd.sjoin(gpd.GeoDataFrame(
                      df,
                      geometry = gpd.points_from_xy(df["longitude"], 
                                                    df["latitude"]), 
                      crs = "EPSG:4326"
                      ),
                  gpd.read_file("Census-Tracts-2020-Data.geojson")
                     .rename(columns = {"GEOID": "census_tract_geoid"})
                     [["geometry", "census_tract_geoid"]],
                  how = "left", predicate = "within")
           .drop(columns = ["geometry", "index_right"]) 
        )
    
    df = pl.from_pandas(df)
                
    return df

def prepare_train_test_data(df, property, model = None):
    """
    Return two DataFrames and two Series for a year-based train-test split, with
    all model-specific processing applied after splitting the DataFrame.
    """
    df = df.with_columns(
        pl.col("price").log().alias("log_price")                
        )
    
    if property == "house":    
        df = df.drop("rooms", "stories", "price", "sale_num", "struct_d", 
                     "extwall_d", "roof_d", "floor_d", "kitchens", "fireplaces", 
                     "age")                  
    elif property == "condo":  
        df = df.drop("rooms", "ac", "price", "land_area", "rmdl")
     
    cat_cols = ["heat_d", "ward", "high_school"]
    
    encoder = OneHotEncoder(
        drop = "first" if model == "lr" else None,
        sparse_output = False,
        handle_unknown = "ignore"
        )
        
    x_train = df.filter(pl.col("sale_year").is_in([2021, 2022, 2023, 2024])) 
    x_train = (
        x_train.with_columns(
            pl.DataFrame(encoder.fit_transform(x_train.select(cat_cols)),
                         schema = encoder.get_feature_names_out(cat_cols)
                                         .tolist())
            )
               .rename(lambda col: col.lower().replace(" ", "_")
                                              .replace("-", "_"))
               .drop(cat_cols + ["log_price"])
        )
    
    x_test = df.filter(pl.col("sale_year") == 2025)
    x_test = (
        x_test.with_columns(
            pl.DataFrame(encoder.transform(x_test.select(cat_cols)), 
                         schema = encoder.get_feature_names_out(cat_cols) 
                                         .tolist())
            )
              .rename(lambda col: col.lower().replace(" ", "_")
                                             .replace("-", "_"))
              .drop(cat_cols + ["log_price"]) 
        )
    
    y_train = (
        df.filter(pl.col("sale_year").is_in([2021, 2022, 2023, 2024]))
          .select("log_price")
          .to_series()  
        )
     
    y_test = (
        df.filter(pl.col("sale_year") == 2025)
          .select("log_price")
          .to_series()
        )
    
    if model == "svm":
        if property == "house":
            scale_cols = ["bedrms", "gba", "land_area", "sale_year", 
                          "ttl_bathrms", "latitude", "longitude", 
                          "commute_walk", "private_wage_workers", 
                          "per_capita_income", "housing_built_pre_1940", 
                          "owner_occupied_housing"]                    
        elif property == "condo":
            scale_cols = ["bedrms", "sale_num", "unit_gba", "sale_year", "age", 
                          "ttl_bathrms", "fireplace", "latitude", "longitude", 
                          "commute_walk", "mean_nonfamily_income", 
                          "housing_built_1950_1959", 
                          "median_owner_cost_mortgage"]
                   
        scaler = StandardScaler()
        
        x_train = x_train.with_columns(
            pl.DataFrame(scaler.fit_transform(x_train.select(scale_cols)), 
                         schema = scale_cols)
            )
        
        x_test = x_test.with_columns(
            pl.DataFrame(scaler.transform(x_test.select(scale_cols)), 
                         schema = scale_cols)
            )
            
    return x_train, x_test, y_train, y_test

def model_metrics_summary(model, y_test, y_pred):
    """
    Return a DataFrame containing the performance and error metrics computed 
    from the machine learning model's predictions on the test DataFrame.
    """
    df = pl.DataFrame({
        "Model": model,
        "R\u00b2": format(r2_score(y_test, y_pred), ".3f"),
        "MAE": "$" + format(
            mean_absolute_error(y_test.exp(), pl.Series(y_pred).exp()),
            ",.0f"),
        "RMSE": "$" + format(
            root_mean_squared_error(y_test.exp(), pl.Series(y_pred).exp()),
            ",.0f"),
        "MdAPE": format(
            median_absolute_percentage_error(y_test.exp(), 
                                             pl.Series(y_pred).exp()),
            ".2%")
        })
    
    return df 
                                                                     
# Part 3: Data Preprocessing

# Section 3.1: Houses

# Create the houses DataFrame containing the appropriate records and columns
houses = (
    pl.read_parquet("DC-CAMA-House-Data.parquet")
      .rename(str.lower)
      .rename({"bathrm": "bathrms",
               "hf_bathrm": "hf_bathrms",
               "bedrm": "bedrms",
               "saledate": "sale_date",
               "intwall_d": "floor_d", 
               "usecode": "use_code",
               "landarea": "land_area"}) 
      .with_columns(
          pl.col("sale_date").str.slice(0, 10).str.to_date("%Y/%m/%d").dt.year()   
            .alias("sale_year"),
          pl.col("qualified").str.strip_chars()
          )
      .filter((pl.col("num_units") == 1) & 
              pl.col("sale_year").is_between(2021, 2025) & 
              (pl.col("sale_year") >= pl.col("ayb")) &
              pl.col("ayb").is_between(1900, 2025) & 
              (((pl.col("sale_year") >= pl.col("yr_rmdl")) & 
                (pl.col("yr_rmdl") >= pl.col("ayb"))) | 
               pl.col("yr_rmdl").is_null()) & 
              (pl.col("qualified") == "Q") & 
              (pl.col("bldg_num") == 1) &
              ((pl.col("struct_d").is_in(["Row End", "Row Inside"]) & 
                (pl.col("use_code") == 11)) |
               ((pl.col("struct_d") == "Single") & (pl.col("use_code") == 12)) |
               ((pl.col("struct_d") == "Semi-Detached") & 
                (pl.col("use_code") == 13))))
      .with_columns(    
          pl.col("ssl").str.replace_all(r"\s+", " "),
          (pl.col("sale_year") - pl.col("ayb")).alias("age"),
          (pl.col("yr_rmdl").is_not_null()).cast(pl.Int8).alias("rmdl")
          )
      .drop("objectid", "heat", "num_units", "ayb", "yr_rmdl", "eyb", 
            "qualified", "bldg_num", "style", "struct", "grade", "grade_d", 
            "cndtn", "cndtn_d", "extwall", "roof", "intwall", "use_code", 
            "gis_last_mod_dttm")          
    )

# Update the houses DataFrame retaining the relevant records and columns
houses = (
    houses.filter(pl.col("bathrms").is_between(1, 4) & 
                  pl.col("hf_bathrms").is_between(0, 2) &
                  pl.col("heat_d").is_in(["Forced Air", "Hot Water Rad", 
                                          "Warm Cool"]) &
                  ~((pl.col("heat_d") == "Warm Cool") & (pl.col("ac") == "N")) &
                  pl.col("rooms").is_between(4, 12) & 
                  (pl.col("rooms") >= pl.col("bedrms")) &
                  pl.col("bedrms").is_between(2, 6) &
                  (((pl.col("stories") == 1) & 
                    (pl.col("style_d") == "1 Story")) |
                   ((pl.col("stories") == 1.5) & 
                    (pl.col("style_d") == "1.5 Story Fin")) |
                   ((pl.col("stories") == 2) & 
                    (pl.col("style_d") == "2 Story")) |
                   ((pl.col("stories") == 2.5) & 
                    (pl.col("style_d") == "2.5 Story Fin")) | 
                   ((pl.col("stories") == 3) & 
                    (pl.col("style_d") == "3 Story"))) &
                  pl.col("price").is_between(300_000, 3_250_000) &
                  pl.col("sale_num").is_between(1, 6) &
                  pl.col("gba").is_between(700, 6_000) &
                  pl.col("extwall_d").is_in(["Brick/Siding", "Common Brick", 
                                             "Vinyl Siding", "Wood Siding"]) &
                  pl.col("roof_d").is_in(["Built Up", "Comp Shingle", 
                                          "Metal- Sms", "Slate"]) &
                  pl.col("floor_d").is_in(["Hardwood", "Hardwood/Carp", 
                                           "Wood Floor"]) & 
                  pl.col("kitchens").is_between(1, 2) & 
                  pl.col("fireplaces").is_between(0, 3) &
                  pl.col("land_area").is_between(400, 15_000))
          .with_columns(
              (pl.col("bathrms") + (pl.col("hf_bathrms") * 0.5))
                 .alias("ttl_bathrms"), 
              pl.when(pl.col("heat_d") == "Hot Water Rad")      
                .then(pl.lit("Hot Water Radiator")) 
                .when(pl.col("heat_d") == "Warm Cool")
                .then(pl.lit("Dual")) 
                .otherwise("heat_d") 
                .alias("heat_d"),
              (pl.col("ac") == "Y").cast(pl.Int8),
              pl.when(pl.col("extwall_d") == "Brick/Siding")
                .then(pl.lit("Brick and Siding"))
                .when(pl.col("extwall_d") == "Common Brick")
                .then(pl.lit("Brick"))
                .when(pl.col("extwall_d") == "Vinyl Siding") 
                .then(pl.lit("Vinyl"))
                .when(pl.col("extwall_d") == "Wood Siding")
                .then(pl.lit("Wood"))
                .alias("extwall_d"),
              pl.when(pl.col("roof_d") == "Built Up") 
                .then(pl.lit("Built-Up"))
                .when(pl.col("roof_d") == "Comp Shingle")
                .then(pl.lit("Composition Shingle"))
                .when(pl.col("roof_d") == "Metal- Sms") 
                .then(pl.lit("Metal")) 
                .otherwise("roof_d")
                .alias("roof_d"),
              pl.when(pl.col("floor_d") == "Hardwood/Carp") 
                .then(pl.lit("Hardwood and Carpet")) 
                .otherwise("floor_d") 
                .alias("floor_d")             
              )
          .drop("bathrms", "hf_bathrms", "style_d")  
    )

# Create the addresses DataFrame containing the appropriate records and columns
addresses_h = (
    pl.read_parquet("DC-Address-Points-Data.parquet")
      .rename(str.lower)
      .select("ssl", "ward", "latitude", "longitude")
      .filter(pl.col("ssl").is_not_null())
      .with_columns(
          pl.col("ssl").str.replace_all(r"\s+", " "),
          pl.col("ward").str.strip_prefix("Ward ")         
          )
      .unique("ssl")
    )
    
# Update the houses DataFrame by performing an inner join with the DataFrames 
houses = houses.join(addresses_h, on = "ssl", how = "inner")

# Update the houses DataFrame by concatenating the DataFrames
houses = (
    pl.concat([houses.pipe(assign_high_school_zone, [2021, 2022, 2023],
                           "DC-High-School-Attendance-Zones-1-Data.geojson"),
               houses.pipe(assign_high_school_zone, [2024, 2025],
                           "DC-High-School-Attendance-Zones-2-Data.geojson")])
    )

# Create the acs DataFrame containing the appropriate records and columns
acs_h = (
    pl.read_parquet("ACS-5-Year-Estimates-Data.parquet")
      .rename(str.lower)
      .rename({"geoid": "census_tract_geoid",
               "dp03_0022e": "commute_walk",
               "dp03_0047e": "private_wage_workers",
               "dp03_0088e": "per_capita_income",
               "dp04_0026e": "housing_built_pre_1940",
               "dp04_0046e": "owner_occupied_housing"})
      .select("census_tract_geoid", "commute_walk", "private_wage_workers", 
              "per_capita_income", "housing_built_pre_1940", 
              "owner_occupied_housing")
      .with_columns(
          pl.col("census_tract_geoid").cast(pl.Utf8)        
          )       
    )

# Update the houses DataFrame by performing a left join with the DataFrames 
houses = (
    houses.pipe(assign_census_tract)
          .join(acs_h, on = "census_tract_geoid", how = "left")
          .sort("sale_date")
          .drop("ssl", "sale_date", "census_tract_geoid")          
    )

# Section 3.2: Condominiums

# Create the condos DataFrame containing the appropriate records and columns
condos = (
    pl.read_parquet("DC-CAMA-Condominium-Data.parquet")
      .rename(str.lower)
      .rename({"bedrm": "bedrms", 
               "bathrm": "bathrms",       
               "hf_bathrm": "hf_bathrms",
               "saledate": "sale_date",
               "living_gba": "unit_gba",
               "usecode": "use_code",
               "landarea": "land_area"})
      .with_columns(
          pl.col("sale_date").str.slice(0, 10).str.to_date("%Y/%m/%d").dt.year()   
            .alias("sale_year"),
          pl.col("qualified").str.strip_chars()
          )
      .filter(pl.col("sale_year").is_between(2021, 2025) &
              (pl.col("sale_year") >= pl.col("ayb")) &
              pl.col("ayb").is_between(1900, 2025) & 
              (((pl.col("sale_year") >= pl.col("yr_rmdl")) & 
                (pl.col("yr_rmdl") >= pl.col("ayb"))) | 
               pl.col("yr_rmdl").is_null()) &
              (pl.col("qualified") == "Q") &
              pl.col("use_code").is_in([16, 17]))
      .with_columns(       
          pl.col("ssl").str.replace_all(r"\s+", " "),
          (pl.col("sale_year") - pl.col("ayb")).alias("age"),
          (pl.col("yr_rmdl").is_not_null()).cast(pl.Int8).alias("rmdl"),
          pl.col("fireplaces").str.strip_chars().cast(pl.Int16)
          ) 
      .drop("bldg_num", "cmplx_num", "ayb", "yr_rmdl", "eyb", "heat", 
            "qualified", "use_code", "gis_last_mod_dttm", "objectid")   
    )

# Update the condos DataFrame retaining the relevant records and columns
condos = (
    condos.filter(pl.col("rooms").is_between(2, 6) &
                  (pl.col("rooms") >= pl.col("bedrms")) &
                  pl.col("bedrms").is_between(0, 3) &
                  pl.col("bathrms").is_between(1, 3) &
                  pl.col("hf_bathrms").is_between(0, 1) &
                  pl.col("heat_d").is_in(["Forced Air", "Hot Water Rad", 
                                          "Ht Pump", "Warm Cool"]) &
                  ~(pl.col("heat_d").is_in(["Ht Pump", "Warm Cool"]) & 
                    (pl.col("ac") == "N")) &
                  pl.col("ac").is_in(["N", "Y"]) &
                  pl.col("fireplaces").is_between(0, 1) &
                  pl.col("price").is_between(150_000, 1_750_000) &
                  pl.col("sale_num").is_between(1, 6) & 
                  pl.col("unit_gba").is_between(500, 1_800) & 
                  pl.col("land_area").is_between(0, 1_500)) 
          .with_columns(
              (pl.col("bathrms") + (pl.col("hf_bathrms") * 0.5))    
                 .alias("ttl_bathrms"),
              pl.when(pl.col("heat_d") == "Hot Water Rad")
                .then(pl.lit("Hot Water Radiator"))
                .when(pl.col("heat_d") == "Ht Pump")
                .then(pl.lit("Heat Pump"))
                .when(pl.col("heat_d") == "Warm Cool")
                .then(pl.lit("Dual"))
                .otherwise("heat_d")
                .alias("heat_d"),
              (pl.col("ac") == "Y").cast(pl.Int8),
              pl.col("fireplaces").cast(pl.Int8).alias("fireplace")               
              )
          .drop("bathrms", "hf_bathrms", "fireplaces")
    )

# Create the addresses DataFrame containing the appropriate records and columns
addresses_c = (
    pl.read_parquet("DC-Address-Points-Data.parquet")
      .rename(str.lower)
      .select("mar_id", "ward", "latitude", "longitude")
      .with_columns(
          pl.col("ward").str.strip_prefix("Ward ")
          )
      .join(pl.read_parquet("DC-Address-Residential-Units-Data.parquet")
              .rename(str.lower)
              .rename({"condo_ssl": "ssl"})
              .select("mar_id", "ssl")
              .filter(pl.col("ssl").is_not_null())
              .with_columns(
                  pl.col("ssl").str.replace_all(r"\s+", " ")
                  )
              .unique("ssl"),
            on = "mar_id", how = "inner")
      .drop("mar_id")  
    )

# Update the condos DataFrame by performing an inner join with the DataFrames              
condos = condos.join(addresses_c, on = "ssl", how = "inner")

# Update the condos DataFrame by concatenating the DataFrames
condos = (
    pl.concat([condos.pipe(assign_high_school_zone, [2021, 2022, 2023],
                           "DC-High-School-Attendance-Zones-1-Data.geojson"),
               condos.pipe(assign_high_school_zone, [2024, 2025],
                           "DC-High-School-Attendance-Zones-2-Data.geojson")])
    )

# Create the acs DataFrame containing the appropriate records and columns
acs_c = (
    pl.read_parquet("ACS-5-Year-Estimates-Data.parquet")
      .rename(str.lower)
      .rename({"geoid": "census_tract_geoid",
               "dp03_0022e": "commute_walk",
               "dp03_0091e": "mean_nonfamily_income",         
               "dp04_0024e": "housing_built_1950_1959",
               "dp04_0101e": "median_owner_cost_mortgage"})
      .select("census_tract_geoid", "commute_walk", "mean_nonfamily_income", 
              "housing_built_1950_1959", "median_owner_cost_mortgage") 
      .with_columns(
          pl.col("census_tract_geoid").cast(pl.Utf8)
          )     
    )
    
# Update the condos DataFrame by performing a left join with the DataFrames    
condos = (
    condos.pipe(assign_census_tract)
          .join(acs_c, on = "census_tract_geoid", how = "left")
          .sort("sale_date")
          .drop("ssl", "sale_date", "census_tract_geoid")
    )   
                    
# Part 4: Exploratory Data Analysis

# Section 4.1: Houses

# Create the descriptive statistics DataFrame
descriptive_stats_h = (
    houses.select(pl.selectors.numeric())
          .drop("latitude", "longitude")
          .describe()
          .filter(pl.col("statistic").is_in(["mean", "min", "50%", "max"]))
          .with_columns(
              pl.col("statistic").replace("50%", "median")
                                 .cast(pl.Enum(["min", "median", "mean", 
                                                "max"])),                               
              pl.selectors.numeric().round(2)
              )
          .sort("statistic")
    )

# Create the Pearson correlation coefficients DataFrame
pearson_corrs_h = (
    pl.concat([pl.DataFrame({
                   "variable": col,
                   "correlation": round(
                       houses.select(pl.corr(col, 
                                             pl.col("price").log())).item(),
                       2)
                   })
               for col in houses.select(pl.selectors.numeric()).drop("price") 
                                .columns])
      .sort(pl.col("correlation").abs(), descending = True)     
    )

# Create the correlation ratios DataFrame 
corr_ratios_h = (
    pl.concat([pl.DataFrame({
                   "variable": col,
                   "correlation_ratio": round(
                       correlation_ratio(houses[col],
                                         houses.select(pl.col("price").log()) 
                                               .to_series()),
                       2)
                   })
               for col in houses.select(pl.selectors.string()).columns])  
      .sort("correlation_ratio", descending = True)         
    )

# Section 4.2: Condominiums

# Create the descriptive statistics DataFrame
descriptive_stats_c = (
    condos.drop("heat_d", "ward", "latitude", "longitude", "high_school")
          .describe()
          .filter(pl.col("statistic").is_in(["mean", "min", "50%", "max"]))
          .with_columns(
              pl.col("statistic").replace("50%", "median")
                                 .cast(pl.Enum(["min", "median", "mean", 
                                                "max"])),                                           
              pl.selectors.numeric().round(2)
              )
          .sort("statistic")
    )

# Create the Pearson correlation coefficients DataFrame
pearson_corrs_c = (
    pl.concat([pl.DataFrame({
                   "variable": col,
                   "correlation": round(
                       condos.select(pl.corr(col,
                                             pl.col("price").log())).item(),
                       2)
                   })
               for col in condos.drop("heat_d", "price", "ward", "high_school")
                                .columns])
      .sort(pl.col("correlation").abs(), descending = True)           
    )

# Create the correlation ratios DataFrame
corr_ratios_c = (
    pl.concat([pl.DataFrame({
                   "variable": col,
                   "correlation_ratio": round(
                       correlation_ratio(condos[col],
                                         condos.select(pl.col("price").log())
                                               .to_series()), 
                       2)
                   })
               for col in condos.select(pl.selectors.string()).columns])
      .sort("correlation_ratio", descending = True)            
    )
                                                          
# Part 5: Machine Learning Models

# Section 5.1: Houses

# Perform the train-test split for the model
x_lr_train_h, x_lr_test_h, y_lr_train_h, y_lr_test_h = (
    houses.pipe(prepare_train_test_data, "house", "lr")
    )

# Fit the model to the training data
lr_fit_h = LinearRegression().fit(x_lr_train_h, y_lr_train_h)

# Perform the train-test split for the model
x_svm_train_h, x_svm_test_h, y_svm_train_h, y_svm_test_h = (
    houses.pipe(prepare_train_test_data, "house", "svm")
    )

# Tune hyperparameters with cross-validation to find the best hyperparameters
svm_best_hp_h = GridSearchCV(
    estimator = SVR(),
    param_grid = {"C": [0.6, 0.7, 0.8], 
                  "epsilon": [0.04, 0.05, 0.06]},
    scoring = "neg_root_mean_squared_error",
    cv = TimeSeriesSplit(n_splits = 5)
    ).fit(x_svm_train_h, y_svm_train_h).best_params_

# Fit the model to the training data
svm_fit_h = SVR(
    C = svm_best_hp_h["C"], 
    epsilon = svm_best_hp_h["epsilon"] 
    ).fit(x_svm_train_h, y_svm_train_h)

# Perform the train-test split for the models
x_tree_train_h, x_tree_test_h, y_tree_train_h, y_tree_test_h = (
    houses.pipe(prepare_train_test_data, "house")
    )

# Tune hyperparameters with cross-validation to find the best hyperparameters
rf_best_hp_h = GridSearchCV(
    estimator = RandomForestRegressor(n_estimators = 500,
                                      random_state = 123),
    param_grid = {"max_depth": [19, 20, 21], 
                  "min_samples_leaf": [1, 2]},
    scoring = "neg_root_mean_squared_error",
    cv = TimeSeriesSplit(n_splits = 5)
    ).fit(x_tree_train_h, y_tree_train_h).best_params_

# Fit the model to the training data
rf_fit_h = RandomForestRegressor(  
    n_estimators = 500,
    max_depth = rf_best_hp_h["max_depth"],
    min_samples_leaf = rf_best_hp_h["min_samples_leaf"],  
    random_state = 123
    ).fit(x_tree_train_h, y_tree_train_h)

# Tune hyperparameters with cross-validation to find the best hyperparameters
lgbm_best_hp_h = GridSearchCV(
    estimator = LGBMRegressor(n_estimators = 1_000, 
                              random_state = 123, 
                              n_jobs = 1,
                              deterministic = True, 
                              verbosity = -1),
    param_grid = {"num_leaves": [34, 35, 36],
                  "learning_rate": [0.005, 0.01, 0.02],
                  "min_child_samples": [35, 36, 37]},
    scoring = "neg_root_mean_squared_error",
    cv = TimeSeriesSplit(n_splits = 5)  
    ).fit(x_tree_train_h, y_tree_train_h).best_params_ 

# Fit the model to the training data
lgbm_fit_h = LGBMRegressor(
    num_leaves = lgbm_best_hp_h["num_leaves"],
    learning_rate = lgbm_best_hp_h["learning_rate"],
    n_estimators = 1_000, 
    min_child_samples = lgbm_best_hp_h["min_child_samples"],
    random_state = 123, 
    n_jobs = 1, 
    deterministic = True, 
    ).fit(x_tree_train_h, y_tree_train_h) 

# Tune hyperparameters with cross-validation to find the best hyperparameters
xgb_best_hp_h = GridSearchCV(
    estimator = XGBRegressor(n_estimators = 500,
                             n_jobs = 1,
                             random_state = 123),
    param_grid = {"max_depth": [3, 4, 5],
                  "learning_rate": [0.04, 0.05, 0.06],
                  "min_child_weight": [0.1, 0.2]},
    scoring = "neg_root_mean_squared_error",
    cv = TimeSeriesSplit(n_splits = 5)
    ).fit(x_tree_train_h, y_tree_train_h).best_params_

# Fit the model to the training data
xgb_fit_h = XGBRegressor( 
    n_estimators = 500,  
    max_depth = xgb_best_hp_h["max_depth"], 
    learning_rate = xgb_best_hp_h["learning_rate"],
    n_jobs = 1,
    min_child_weight = xgb_best_hp_h["min_child_weight"], 
    random_state = 123
    ).fit(x_tree_train_h, y_tree_train_h) 

# Create the performance and error metrics DataFrame
model_metrics_h = pl.concat(
    model_metrics_summary(model, y_test, y_pred)
    for model, y_test, y_pred in [  
    ("Linear Regression", y_lr_test_h, lr_fit_h.predict(x_lr_test_h)),
    ("Support Vector Machine", y_svm_test_h, svm_fit_h.predict(x_svm_test_h)),
    ("Random Forest", y_tree_test_h, rf_fit_h.predict(x_tree_test_h)),
    ("LightGBM", y_tree_test_h, lgbm_fit_h.predict(x_tree_test_h)), 
    ("XGBoost", y_tree_test_h, xgb_fit_h.predict(x_tree_test_h))
    ]
    )

# Create the SHAP feature importance DataFrame 
shap_importance_h = (
    pl.DataFrame(shap.TreeExplainer(lgbm_fit_h)
                     .shap_values(x_tree_train_h.to_numpy()), 
                 schema = x_tree_train_h.columns)
      .select(pl.all().abs())
      .with_columns(
          [pl.sum_horizontal(pl.selectors.starts_with(prefix)).alias(prefix)
           for prefix in ["heat_d", "ward", "high_school"]]
          )
      .drop(pl.selectors.starts_with(("heat_d_", "ward_", "high_school_")))
      .select(pl.all().mean().round(2))
      .transpose(include_header = True, 
                 header_name = "predictor", 
                 column_names = ["mean_abs_shap"])
      .sort("mean_abs_shap", descending = True)
    )
                          
# Section 5.2: Condominiums

# Perform the train-test split for the model
x_lr_train_c, x_lr_test_c, y_lr_train_c, y_lr_test_c = (
    condos.pipe(prepare_train_test_data, "condo", "lr")
    )

# Fit the model to the training data
lr_fit_c = LinearRegression().fit(x_lr_train_c, y_lr_train_c)

# Perform the train-test split for the model
x_svm_train_c, x_svm_test_c, y_svm_train_c, y_svm_test_c = (
    condos.pipe(prepare_train_test_data, "condo", "svm")
    )

# Tune hyperparameters with cross-validation to find the best hyperparameters
svm_best_hp_c = GridSearchCV(
    estimator = SVR(), 
    param_grid = {"C": [0.6, 0.7, 0.8], 
                  "epsilon": [0.04, 0.05, 0.06]},
    scoring = "neg_root_mean_squared_error", 
    cv = TimeSeriesSplit(n_splits = 5)
    ).fit(x_svm_train_c, y_svm_train_c).best_params_
 
# Fit the model to the training data
svm_fit_c = SVR(
    C = svm_best_hp_c["C"], 
    epsilon = svm_best_hp_c["epsilon"] 
    ).fit(x_svm_train_c, y_svm_train_c)

# Perform the train-test split for the models
x_tree_train_c, x_tree_test_c, y_tree_train_c, y_tree_test_c = (
    condos.pipe(prepare_train_test_data, "condo")
    ) 

# Tune hyperparameters with cross-validation to find the best hyperparameters
rf_best_hp_c = GridSearchCV(
    estimator = RandomForestRegressor(n_estimators = 500,
                                      random_state = 123),
    param_grid = {"max_depth": [18, 19, 20], 
                  "min_samples_leaf": [1, 2]},
    scoring = "neg_root_mean_squared_error",
    cv = TimeSeriesSplit(n_splits = 5)
    ).fit(x_tree_train_c, y_tree_train_c).best_params_

# Fit the model to the training data
rf_fit_c = RandomForestRegressor(  
    n_estimators = 500,
    max_depth = rf_best_hp_c["max_depth"],
    min_samples_leaf = rf_best_hp_c["min_samples_leaf"],  
    random_state = 123
    ).fit(x_tree_train_c, y_tree_train_c)

# Tune hyperparameters with cross-validation to find the best hyperparameters
lgbm_best_hp_c = GridSearchCV(
    estimator = LGBMRegressor(n_estimators = 1_000, 
                              random_state = 123,
                              n_jobs = 1,
                              deterministic = True, 
                              verbosity = -1),
    param_grid = {"num_leaves": [23, 24, 25],
                  "learning_rate": [0.02, 0.03, 0.04],
                  "min_child_samples": [44, 45, 46]},
    scoring = "neg_root_mean_squared_error",
    cv = TimeSeriesSplit(n_splits = 5)
    ).fit(x_tree_train_c, y_tree_train_c).best_params_

# Fit the model to the training data
lgbm_fit_c = LGBMRegressor(
    num_leaves = lgbm_best_hp_c["num_leaves"],
    learning_rate = lgbm_best_hp_c["learning_rate"],
    n_estimators = 1_000, 
    min_child_samples = lgbm_best_hp_c["min_child_samples"],
    random_state = 123,
    n_jobs = 1, 
    deterministic = True, 
    ).fit(x_tree_train_c, y_tree_train_c)

# Tune hyperparameters with cross-validation to find the best hyperparameters
xgb_best_hp_c = GridSearchCV(
    estimator = XGBRegressor(n_estimators = 500,
                             n_jobs = 1, 
                             random_state = 123),
    param_grid = {"max_depth": [3, 4, 5], 
                  "learning_rate": [0.06, 0.07, 0.08], 
                  "min_child_weight": [0.1, 0.2]},
    scoring = "neg_root_mean_squared_error",
    cv = TimeSeriesSplit(n_splits = 5)
    ).fit(x_tree_train_c, y_tree_train_c).best_params_

# Fit the model to the training data
xgb_fit_c = XGBRegressor( 
    n_estimators = 500,  
    max_depth = xgb_best_hp_c["max_depth"], 
    learning_rate = xgb_best_hp_c["learning_rate"],
    n_jobs = 1,
    min_child_weight = xgb_best_hp_c["min_child_weight"], 
    random_state = 123
    ).fit(x_tree_train_c, y_tree_train_c) 

# Create the performance and error metrics DataFrame
model_metrics_c = pl.concat(
    model_metrics_summary(model, y_test, y_pred)
    for model, y_test, y_pred in [    
    ("Linear Regression", y_lr_test_c, lr_fit_c.predict(x_lr_test_c)),
    ("Support Vector Machine", y_svm_test_c, svm_fit_c.predict(x_svm_test_c)),
    ("Random Forest", y_tree_test_c, rf_fit_c.predict(x_tree_test_c)),
    ("LightGBM", y_tree_test_c, lgbm_fit_c.predict(x_tree_test_c)), 
    ("XGBoost", y_tree_test_c, xgb_fit_c.predict(x_tree_test_c))
    ]
    )

# Create the SHAP feature importance DataFrame
shap_importance_c = (
    pl.DataFrame(shap.TreeExplainer(lgbm_fit_c)
                     .shap_values(x_tree_train_c.to_numpy()),
                 schema = x_tree_train_c.columns)
      .select(pl.all().abs())
      .with_columns(
          [pl.sum_horizontal(pl.selectors.starts_with(prefix)).alias(prefix)
           for prefix in ["heat_d", "ward", "high_school"]]
          )
      .drop(pl.selectors.starts_with(("heat_d_", "ward_", "high_school_")))
      .select(pl.all().mean().round(2))
      .transpose(include_header = True, 
                 header_name = "predictor", 
                 column_names = ["mean_abs_shap"])
      .sort("mean_abs_shap", descending = True)
    )    