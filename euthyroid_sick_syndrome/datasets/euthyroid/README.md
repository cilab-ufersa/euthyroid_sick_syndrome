## Euthyroid sick syndrome dataset 

### Original Features:

| age| sex| on_thyroxine | query_on_thyroxine | on_antithyroid_medication | thyroid_surgery | query_hypothyroid | query_hyperthyroid | pregnant| sick| tumor | lithium| goitre|TSH_measured| TSH | T3_measured | T3 | TT4_measured | TT4 | T4U_measured | T4U | FTI_measured | FTI | TBG_measured | TBG | 
| ----| ----| ---- | ---- |----| ---- | ----| ---- | ----| ----| ---- |----|----|----| ---- | ---- | ---- | ---- | ----| ---- |---- | ---- | ---- | ----| ----| 

### After Preprocessing:

| age| sex| on_thyroxine | query_on_thyroxine | on_antithyroid_medication | thyroid_surgery | query_hypothyroid | query_hyperthyroid | pregnant| sick| tumor | lithium| goitre| TSH | T3 | TT4 |  T4U |  FTI | 
| ----| ----| ---- | ---- |----| ---- | ----| ---- | ----| ----| ---- |----|----|----| ---- | ---- | ---- | ---- |  

### considering only blood exams features

| age| sex| TSH | T3 | TT4 |  T4U |  FTI | 
| ----| ----| ---- | ---- |----| ---- | ----| 

where 

- Levothyroxine  (T4 /T4U)
- Triiodothyronine  (T3)
- Total  T4 (TT4)
- Free  T4  Index  (FTI) 
- Thyroid  Stimulating  Hormone  (TSH)


