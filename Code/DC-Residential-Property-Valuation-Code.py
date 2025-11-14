# Title: DC Residential Property Valuation Analysis
# Author: Alexander Zakrzeski
# Date: November 13, 2025

# Part 1: Setup and Configuration

# Load to import, clean, and wrangle data
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
          )
    )

address_points = pl.read_parquet("DC-Address-Points-Data.parquet")
address_points = address_points.rename(str.lower)
address_points = address_points.rename({"zipcode": "zip_code"})
address_points = address_points.select("ward", "zip_code", "quadrant", "anc", 
                                       "smd", "latitude", "longitude", "ssl")
address_points = address_points.drop_nulls("ssl")
address_points = address_points.with_columns(pl.col("ward").str.replace(r"^Ward ", "").cast(pl.Int64).alias("ward"))
address_points = address_points.with_columns(pl.col("zip_code").cast(pl.Utf8).alias("zip_code"))



# Load the data from the CSV file
addresses = pd.read_csv("DC-Addresses-Data.csv", usecols = ["WARD", "SSL"])

# Rename columns, drop rows with missing values, and modify values of columns
addresses = addresses.rename(columns = str.lower).dropna()
addresses["ward"] = addresses["ward"].str.replace(r"^Ward ", "", regex = True)
addresses["ssl"] = addresses["ssl"].str.replace(r"\s{2,}", " ", regex = True)

# Drop duplicates, reset the index, and reorder the column
addresses = addresses.drop_duplicates().reset_index(drop = True)
addresses.insert(0, "ssl", addresses.pop("ssl"))

# Perform a left join, drop rows with missing values, and drop columns
appraisals = (appraisals.
  merge(addresses, on = "ssl", how = "left").
  dropna(subset = "ward").
  drop(columns = ["ssl", "bathrms", "hf_bathrms", "ayb", "yr_rmdl", 
                  "qualified", "sale_num", "gba", "bldg_num", "style_d", 
                  "struct_d", "usecode", "saledate_ym"])) 

# Reorder the columns, sort the rows in ascending order, and reset the index
appraisals.insert(0, "saledate", appraisals.pop("saledate"))
appraisals.insert(1, "saledate_y", appraisals.pop("saledate_y"))
appraisals.insert(2, "ward", appraisals.pop("ward"))
appraisals.insert(3, "age", appraisals.pop("age"))
appraisals.insert(4, "rmdl", appraisals.pop("rmdl"))
appraisals.insert(5, "ttl_bathrms", appraisals.pop("ttl_bathrms"))
appraisals.insert(19, "price", appraisals.pop("price"))
appraisals.insert(20, "log_price", appraisals.pop("log_price"))
appraisals.insert(12, "log_gba", appraisals.pop("log_gba"))
appraisals = appraisals.sort_values(by = "saledate").reset_index(drop = True)