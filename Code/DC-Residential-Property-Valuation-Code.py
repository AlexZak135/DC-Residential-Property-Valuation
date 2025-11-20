# Title: DC Residential Property Valuation Analysis
# Author: Alexander Zakrzeski
# Date: November 19, 2025

# Part 1: Setup and Configuration

# Load to import, clean, and wrangle data
import geopandas as gpd
import os
import polars as pl

# Set the working directory
os.chdir("/Users/atz5/Desktop/DC-Residential-Property-Valuation/Data")

# Part 2: Function Definitions

# Part 3: Data Preprocessing

# Load the data from the Parquet file, rename columns, and drop columns
houses = (
    pl.read_parquet("DC-House-Valuation-Data.parquet")
      .rename(str.lower)
      .rename({"bathrm": "bathrms",
               "hf_bathrm": "hf_bathrms",
               "bedrm": "bedrms",
               "saledate": "sale_date",
               "intwall_d": "floor_d"})
      .drop("objectid", "heat", "eyb", "style", "struct", "grade", "grade_d", 
            "cndtn", "cndtn_d", "extwall", "roof", "intwall", 
            "gis_last_mod_dttm")
      # Filter, modify the values of existing columns, and create new columns
      .filter(pl.col("bathrms").is_between(1, 4) & 
              pl.col("hf_bathrms").is_between(0, 2) &
              pl.col("heat_d").is_in(["Forced Air", "Hot Water Rad", 
                                      "Warm Cool"]) &
              ~(((pl.col("heat_d") == "Forced Air") & (pl.col("ac") == "N")) |
                ((pl.col("heat_d") == "Warm Cool") & (pl.col("ac") == "N"))) &
              pl.col("ac").is_in(["N", "Y"]) &
              (pl.col("num_units") == 1) &
              pl.col("rooms").is_between(4, 12) & 
              (pl.col("rooms") >= pl.col("bedrms")) & 
              pl.col("bedrms").is_between(2, 6) & 
              pl.col("ayb").is_between(1_890, 2_025) & 
              (pl.col("ayb") <= pl.col("sale_date") 
                                  .str.split(" ").list.first()
                                  .str.to_date("%Y/%m/%d").dt.year()) &
              ((pl.col("yr_rmdl").is_between(pl.col("ayb"), 2_025) &
                (pl.col("yr_rmdl") <= pl.col("sale_date")
                                        .str.split(" ").list.first()
                                        .str.to_date("%Y/%m/%d").dt.year())) | 
               pl.col("yr_rmdl").is_null()) &
              (((pl.col("stories") == 1) & (pl.col("style_d") == "1 Story")) |
               ((pl.col("stories") == 1.5) & 
                (pl.col("style_d") == "1.5 Story Fin")) |
               ((pl.col("stories") == 2) & (pl.col("style_d") == "2 Story")) |
               ((pl.col("stories") == 2.5) & 
                (pl.col("style_d") == "2.5 Story Fin")) |
               ((pl.col("stories") == 3) & (pl.col("style_d") == "3 Story"))) &
              (pl.col("sale_date")
                 .str.split(" ").list.first().str.to_date("%Y/%m/%d")
                 .is_between(pl.date(2_019, 1, 1), pl.date(2_025, 5, 9))) &
              pl.col("price").is_between(300_000, 3_250_000) & 
              (pl.col("qualified") == "Q") &
              pl.col("sale_num").is_between(1, 6) &
              pl.col("gba").is_between(700, 6_000) &
              (pl.col("bldg_num") == 1) &
              ((pl.col("struct_d").is_in(["Row End", "Row Inside"]) &
                (pl.col("usecode") == 11)) |
               ((pl.col("struct_d") == "Single") & (pl.col("usecode") == 12)) |
               ((pl.col("struct_d") == "Semi-Detached") & 
                (pl.col("usecode") == 13))) &
              pl.col("extwall_d").is_in(["Brick/Siding", "Common Brick", 
                                         "Stucco", "Vinyl Siding", 
                                         "Wood Siding"]) &
              pl.col("roof_d").is_in(["Built Up", "Comp Shingle", "Metal- Sms", 
                                      "Slate"]) &
              pl.col("floor_d").is_in(["Carpet", "Hardwood", "Hardwood/Carp", 
                                       "Wood Floor"]) &        
              pl.col("kitchens").is_between(1, 2) &
              pl.col("fireplaces").is_between(0, 3) &
              pl.col("landarea").is_between(400, 15_000))
      .with_columns(
          pl.col("ssl").str.replace(r"\s{2,}", " ").alias("ssl"),
          (pl.col("bathrms") + (pl.col("hf_bathrms") * 0.5))
             .alias("ttl_bathrms"),
          pl.when(pl.col("heat_d") == "Hot Water Rad")
            .then(pl.lit("Hot Water Radiator"))
            .when(pl.col("heat_d") == "Warm Cool")
            .then(pl.lit("Dual"))
            .otherwise("heat_d")
            .alias("heat_d"),
          pl.when(pl.col("ac") == "Y")
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No")) 
            .alias("ac"),
          (pl.col("sale_date")       
             .str.split(" ").list.first().str.to_date("%Y/%m/%d").dt.year() - 
           pl.col("ayb"))
             .alias("age"),
          pl.when(pl.col("yr_rmdl").is_not_null())
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No"))
            .alias("rmdl"),
          pl.col("sale_date")
            .str.split(" ").list.first().str.to_date("%Y/%m/%d")        
            .alias("sale_date"),
          pl.col("sale_date") 
            .str.split(" ").list.first()
            .str.to_date("%Y/%m/%d").dt.strftime("%B") 
            .alias("sale_month"),
          pl.when(pl.col("extwall_d") == "Common Brick")
            .then(pl.lit("Brick"))
            .when(pl.col("extwall_d") == "Brick/Siding")
            .then(pl.lit("Brick and Siding"))
            .when(pl.col("extwall_d") == "Vinyl Siding")
            .then(pl.lit("Vinyl"))
            .when(pl.col("extwall_d") == "Wood Siding")  
            .then(pl.lit("Wood"))
            .otherwise("extwall_d")
            .alias("extwall_d"),
          pl.when(pl.col("roof_d") == "Comp Shingle")
            .then(pl.lit("Composition Shingle"))
            .when(pl.col("roof_d") == "Metal- Sms") 
            .then(pl.lit("Metal"))
            .when(pl.col("roof_d") == "Built Up") 
            .then(pl.lit("Built-Up"))
            .otherwise("roof_d")
            .alias("roof_d"),
          pl.when(pl.col("floor_d") == "Hardwood/Carp")
            .then(pl.lit("Hardwood and Carpet"))
            .otherwise("floor_d")
            .alias("floor_d")
     # Perform an inner join with the Parquet file to include new columns
     ).join(pl.read_parquet("DC-Address-Points-Data.parquet")
              .rename(str.lower)
              .select("ssl", "ward", "quadrant", "latitude", "longitude")
              .with_columns(
                  pl.col("ssl").str.replace(r"\s{2,}", " ").alias("ssl"),
                  pl.col("ward").str.replace(r"^Ward ", "").alias("ward"),
                  pl.when(pl.col("quadrant") == "NE")
                    .then(pl.lit("Northeast"))
                    .when(pl.col("quadrant") == "NW")
                    .then(pl.lit("Northwest"))
                    .when(pl.col("quadrant") == "SE")
                    .then(pl.lit("Southeast"))
                    .when(pl.col("quadrant") == "SW")
                    .then(pl.lit("Southwest"))
                    .otherwise(None)
                    .alias("quadrant")                      
             ).unique(["ssl", "ward", "quadrant"]), 
            on = "ssl", how = "inner")
    )
                  
# Perform a spatial left join with the shapefile to include a new column
houses = (
    gpd.sjoin(gpd.GeoDataFrame(
                  houses.to_pandas(),
                  geometry = gpd.points_from_xy(houses.to_pandas()["longitude"], 
                                                houses.to_pandas()["latitude"]), 
                  crs = "EPSG:4326"
                  ),
              gpd.read_file("High-School-Attendance-Zones-Shapefile/"
                            "High-School-Attendance-Zones.shp")
                 .rename(columns = str.lower)
                 [["geometry", "name"]]
                 .to_crs("EPSG:4326"),
              how = "left", predicate = "within")
       .rename(columns = {"name": "high_school"})
       .drop(columns = ["geometry", "index_right"])
       .pipe(pl.from_pandas)
    )
    
################################################################################

os.chdir("/Users/atz5/Desktop")

sat_18_19 = pl.read_excel("School Year 2018-2019 SAT Scores.xlsx")
sat_18_19 = sat_18_19.slice(2).rename(dict(zip(sat_18_19.columns, sat_18_19.row(1)))) 
sat_18_19 = sat_18_19.rename({"School Name": "high_school",
                              "Evidence-Based Reading and Writing Average": "ebrw_avg_sat",
                              "Math Average": "math_avg_sat",
                              "Total Average": "total_avg_sat"})
sat_18_19 = sat_18_19.with_columns(*[pl.col(c).cast(pl.Float64).round().cast(pl.Int64).alias(c)
                                     for c in ["ebrw_avg_sat", "math_avg_sat", "total_avg_sat"]],
                                   pl.lit("2019").alias("year"))
sat_18_19 = sat_18_19.select("year", "high_school", "ebrw_avg_sat", "math_avg_sat", "total_avg_sat")
sat_18_19 = sat_18_19.filter(pl.col("high_school") != "District")

sat_19_20 = pl.read_excel("School-Year-2019-2020-SAT-Scores.xlsx")
sat_19_20 = sat_19_20.slice(2).rename(dict(zip(sat_19_20.columns, sat_19_20.row(1)))) 
sat_19_20 = sat_19_20.rename({"School Name": "high_school",
                              "Evidence-Based Reading and Writing Average": "ebrw_avg_sat",
                              "Math Average": "math_avg_sat",
                              "Total Average": "total_avg_sat"})
sat_19_20 = sat_19_20.with_columns(*[pl.col(c).cast(pl.Int64).alias(c)
                                     for c in ["ebrw_avg_sat", "math_avg_sat", "total_avg_sat"]],
                                   pl.lit("2020").alias("year"))
sat_19_20 = sat_19_20.select("year", "high_school", "ebrw_avg_sat", "math_avg_sat", "total_avg_sat")
sat_19_20 = sat_19_20.filter(pl.col("high_school") != "District")

sat_20_21 = pl.read_excel("SchoolYear2020-2021-SATScores.xlsx")
sat_20_21 = sat_20_21.slice(1).rename(dict(zip(sat_20_21.columns, sat_20_21.row(0))))
sat_20_21 = sat_20_21.rename({"School Name": "high_school",
                              "Evidence-Based Reading and Writing Average": "ebrw_avg_sat",
                              "Math Average": "math_avg_sat",
                              "Total Average": "total_avg_sat"})
sat_20_21 = sat_20_21.with_columns(*[pl.col(c).cast(pl.Int64).alias(c)
                                     for c in ["ebrw_avg_sat", "math_avg_sat", "total_avg_sat"]],
                                   pl.lit("2021").alias("year"))
sat_20_21 = sat_20_21.select("year", "high_school", "ebrw_avg_sat", "math_avg_sat", "total_avg_sat")
sat_20_21 = sat_20_21.filter(pl.col("high_school") != "District")







sat_21_22 = pl.read_excel("School Year 2021-2022 SAT Scores.xlsx")
sat_21_22 = sat_21_22.slice(1).rename(dict(zip(sat_21_22.columns, sat_21_22.row(0))))

sat_22_23 = pl.read_excel("School Year 2022-2023 SAT Scores.xlsx")
sat_22_23 = sat_22_23.slice(1).rename(dict(zip(sat_22_23.columns, sat_22_23.row(0))))

sat_23_24 = pl.read_excel("School Year 2023-2024 SAT Scores_0.xlsx")
sat_23_24 = sat_23_24.slice(1).rename(dict(zip(sat_23_24.columns, sat_23_24.row(0))))

sat_18_24 = pl.concat([sat_18_19, sat_19_20, sat_20_21, sat_21_22, sat_22_23, sat_23_24])

# 2019 houses → 2018–2019 SAT
# 2020 houses → 2019–2020 SAT
# 2021 houses → 2020–2021 SAT
# 2022 houses → 2021–2022 SAT
# 2023 houses → 2022–2023 SAT
# 2024 houses → 2023–2024 SAT
# 2025 houses → 2023–2024 SAT