# Title: DC Residential Property Valuation Analysis
# Author: Alexander Zakrzeski
# Date: November 18, 2025

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
     # Perform an inner join and load the data from the Parquet file
     ).join(pl.read_parquet("DC-Address-Points-Data.parquet")
              # Rename columns and select columns
              .rename(str.lower)
              .select("ssl", "ward", "quadrant", "latitude", "longitude")
              # Modify the values of existing columns and remove duplicates  
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
                
# Perform a spatial left join, rename a column, and drop columns
houses = (
    gpd.sjoin(gpd.GeoDataFrame(
                  houses.to_pandas(),
                  geometry = gpd.points_from_xy(houses.to_pandas()["longitude"], 
                                                houses.to_pandas()["latitude"]), 
                  crs = "EPSG:4326"
                  ),
              gpd.read_file(("High-School-Attendance-Zones-Shapefile/" 
                             "High-School-Attendance-Zones.shp"))
                 .rename(columns = str.lower)
                 [["geometry", "name"]]
                 .to_crs("EPSG:4326"),
              how = "left", predicate = "within")
       .rename(columns = {"name": "high_school"})
       .drop(columns = ["geometry", "index_right"])
       # Convert to a Polars DataFrame
       .pipe(pl.from_pandas)
    )