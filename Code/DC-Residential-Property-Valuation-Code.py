# Title: DC Residential Property Valuation Analysis
# Author: Alexander Zakrzeski
# Date: November 12, 2025

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
      .with_columns(
          pl.col("ssl").str.replace_all(r"\s{2,}", " ").alias("ssl"),
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
            .alias("ac")
          )
    )
                  
                  
 
 
                            
          (pl.col("sale_date")
             .str.split(" ").list.first()
             .str.to_date(format = "%Y/%m/%d")
             .dt.year() - pl.col("ayb"))
             .alias("age"), 
          pl.when(pl.col("yr_rmdl").is_not_null())
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No"))
            .alias("rmdl"),
          pl.col("sale_date")
            .str.split(" ").list.first().str.to_date(format = "%Y/%m/%d")
            .alias("sale_date"),
          pl.col("sale_date")
            .str.split(" ").list.first()
            .str.to_date(format = "%Y/%m/%d")
            .dt.month()
            .map_elements(lambda x: ["January", "February", "March", "April", 
                                     "May", "June", "July", "August", 
                                     "September", "October", "November", 
                                     "December"][x - 1], 
                          return_dtype = pl.Utf8)
            .alias("sale_month"),        
          pl.when(pl.col("style_d") == "1.5 Story Fin")
            .then(pl.lit("1.5 Stories"))
            .when(pl.col("style_d") == "2 Story") 
            .then(pl.lit("2 Stories"))
            .when(pl.col("style_d") == "2.5 Story Fin")
            .then(pl.lit("2.5 Stories"))
            .when(pl.col("style_d") == "3 Story")
            .then(pl.lit("3 Stories"))
            .otherwise("style_d")
            .alias("style_d"),
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


                                                                   
################################################################################
houses.select(pl.col().value_counts())
houses.select(pl.col()).unique().sort()

"ayb"
"yr_rmdl"
"stories"
"sale_date"
"price"
"qualified"
"sale_num"
"gba"
"bldg_num"
"style_d"
"struct_d"
"extwall_d"
"roof_d"
"floor_d"
"kitchens"
"fireplaces"
"usecode"
"landarea"

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
