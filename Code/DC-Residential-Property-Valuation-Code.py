# Title: DC Residential Property Valuation Analysis
# Author: Alexander Zakrzeski
# Date: November 11, 2025

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
              pl.col("heat_d").is_in(["Forced Air", "Hot Water Rad", "Ht Pump", 
                                      "Warm Cool"]) &
              ~(((pl.col("heat_d") == "Forced Air") & (pl.col("ac") == "N")) |
                ((pl.col("heat_d") == "Ht Pump") & (pl.col("ac") == "N")) |
                ((pl.col("heat_d") == "Warm Cool") & (pl.col("ac") == "N"))) &
              pl.col("ac").is_in(["N", "Y"]) &
              (pl.col("num_units") == 1) &
              pl.col("rooms").is_between(4, 12) & 
              (pl.col("rooms") >= pl.col("bedrms")) & 
              pl.col("bedrms").is_between(2, 6) & 
              pl.col("ayb").is_between(1_890, 2_025) & 
              (pl.col("ayb") <= pl.col("sale_date")
                                  .str.split(" ").list.first()             
                                  .str.to_date(format = "%Y/%m/%d")
                                  .dt.year()) &
              ((pl.col("yr_rmdl").is_between(pl.col("ayb"), 2_025) &
                (pl.col("yr_rmdl") <= pl.col("sale_date")
                                        .str.split(" ").list.first()
                                        .str.to_date(format = "%Y/%m/%d")
                                        .dt.year())) | 
               pl.col("yr_rmdl").is_null()) &
              (((pl.col("stories") == 1) & (pl.col("style_d") == "1 Story")) |
               ((pl.col("stories") == 1.5) & 
                (pl.col("style_d") == "1.5 Story Fin")) |
               ((pl.col("stories") == 2) & (pl.col("style_d") == "2 Story")) |
               ((pl.col("stories") == 2.5) & 
                (pl.col("style_d") == "2.5 Story Fin")) |
               ((pl.col("stories") == 3) & (pl.col("style_d") == "3 Story"))) &
              (pl.col("sale_date")
                 .str.split(" ").list.first()
                 .str.to_date(format = "%Y/%m/%d")
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
    )

################################################################################
# Check Data Types and Columns


# Modify the values of existing columns and create new columns
appraisals["ssl"] = appraisals["ssl"].str.replace(r"\s{2,}", " ", regex = True) 
appraisals["ttl_bathrms"] = conditional_map(  
  appraisals["bathrms"] + (appraisals["hf_bathrms"] * 0.5) <= 3.5,
  remove_dot_zero(appraisals["bathrms"] + (appraisals["hf_bathrms"] * 0.5)),
  True, "4 or More"  
  ) 
appraisals["heat_d"] = conditional_map(  
  appraisals["heat_d"] == "Forced Air", "Forced Air",
  appraisals["heat_d"] == "Hot Water Rad", "Hot Water",
  appraisals["heat_d"] == "Warm Cool", "Dual Climate", 
  True, "Other"
  )
appraisals["ac"] = appraisals["ac"].replace({"Y": "Yes", "N": "No"})
appraisals["num_units"] = conditional_map( 
  appraisals["num_units"] == 1, "1", 
  True, "2 or More"
  )
appraisals["rooms"] = conditional_map( 
  appraisals["rooms"] <= 5, "5 or Fewer", 
 (appraisals["rooms"] >= 6) & (appraisals["rooms"] <= 9), 
  remove_dot_zero(appraisals["rooms"]), 
  appraisals["rooms"] >= 10, "10 or More"  
  ) 
appraisals["bedrms"] = conditional_map( 
  appraisals["bedrms"] <= 2, "2 or Fewer", 
 (appraisals["bedrms"] >= 3) & (appraisals["bedrms"] <= 4), 
  remove_dot_zero(appraisals["bedrms"]),
  appraisals["bedrms"] >= 5, "5 or More"
  )
appraisals["age"] = conditional_map(
  pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 10, 
  "10 or Fewer",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 11) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 60)),
  "11 to 60",  
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 61) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 70)), 
  "61 to 70",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 71) & 
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 80)), 
  "71 to 80",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 81) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 90)), 
  "81 to 90",
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 91) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 100)), 
  "91 to 100", 
((pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 101) &
 (pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] <= 110)), 
  "101 to 110", 
  pd.to_datetime(appraisals["saledate"]).dt.year - appraisals["ayb"] >= 111, 
  "111 or More"  
  ) 
appraisals["rmdl"] = conditional_map( 
 ~appraisals["yr_rmdl"].isna(), "Yes",
  True, "No" 
  )
appraisals["stories"] = conditional_map( 
  appraisals["stories"] <= 2, "2 or Fewer",
  True, "2.5 or More"
  )
appraisals["saledate"] = pd.to_datetime(appraisals["saledate"]).dt.date
appraisals["saledate_ym"] = (pd.to_datetime(appraisals["saledate"]). 
                             dt.to_period("M").dt.start_time.dt.date)
appraisals["saledate_y"] = conditional_map( 
  pd.to_datetime(appraisals["saledate"]).dt.year.isin([2023, 2024]), "2023+", 
  True, pd.to_datetime(appraisals["saledate"]).dt.year.astype(str) 
  )
appraisals["log_price"] = np.log(appraisals["price"])
appraisals["log_gba"] = np.log(appraisals["gba"])
appraisals["extwall_d"] = conditional_map( 
  appraisals["extwall_d"].isin(["Brick Veneer", "Brick/Siding", "Brick/Stone", 
                                "Brick/Stucco", "Common Brick", 
                                "Face Brick"]), "Brick", 
  appraisals["extwall_d"].isin(["Shingle", "Stucco", "Vinyl Siding", 
                                "Wood Siding"]), "Siding and Stucco", 
  appraisals["extwall_d"].isin(["Stone", "Stone Veneer", "Stone/Siding", 
                                "Stone/Stucco"]), "Stone" 
  )
appraisals["roof_d"] = conditional_map( 
  appraisals["roof_d"].isin(["Clay Tile", "Slate"]), "Tile",
  appraisals["roof_d"].isin(["Comp Shingle", "Composition Ro", "Shake", 
                             "Shingle"]), "Shingle", 
  appraisals["roof_d"].isin(["Metal- Cpr", "Metal- Pre", 
                             "Metal- Sms"]), "Metal",
  appraisals["roof_d"] == "Built Up", "Flat"  
  )
appraisals["floor_d"] = conditional_map( 
  appraisals["floor_d"].isin(["Carpet", "Vinyl Sheet"]), "Soft", 
  True, "Hard"
  )
appraisals["kitchens"] = conditional_map( 
  appraisals["kitchens"] == 1, "1",   
  True, "2 or More" 
  )
appraisals["fireplaces"] = conditional_map( 
  appraisals["fireplaces"] <= 1, remove_dot_zero(appraisals["fireplaces"]),  
  True, "2 or More" 
  )
appraisals["landarea"] = conditional_map( 
  appraisals["landarea"] <= 999, "999 or Fewer",  
 (appraisals["landarea"] >= 1000) & (appraisals["landarea"] <= 1999),
  "1,000 to 1,999",
 (appraisals["landarea"] >= 2000) & (appraisals["landarea"] <= 4999), 
  "2,000 to 4,999", 
  appraisals["landarea"] >= 5000, "5,000 or More"
  )

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