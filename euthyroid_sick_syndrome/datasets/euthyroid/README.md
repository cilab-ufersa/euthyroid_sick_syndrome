## Euthyroid sick syndrome dataset 

### Original Features:

| age| sex| on_thyroxine | query_on_thyroxine | on_antithyroid_medication | thyroid_surgery | query_hypothyroid | query_hyperthyroid | pregnant| sick| tumor | lithium| goitre|TSH_measured| TSH | T3_measured | T3 | TT4_measured | TT4 | T4U_measured | T4U | FTI_measured | FTI | TBG_measured | TBG | 
| ----| ----| ---- | ---- |----| ---- | ----| ---- | ----| ----| ---- |----|----|----| ---- | ---- | ---- | ---- | ----| ---- |---- | ---- | ---- | ----| ----| 

The Sick-Euthyroid dataset contains the following 27 features for each patient:

Age: patient's age in years.

Sex: patient's gender (Male or Female).

On Thyroxine: indicates if the patient is taking thyroxine medication.

Query on Thyroxine: indicates if the query is related to thyroxine medication.

On Antithyroid Medication: indicates if the patient is taking antithyroid medication.

Sick: indicates if the patient has a thyroid disorder (Yes or No).

Pregnant: indicates if the patient is pregnant (Yes or No).

Thyroid Surgery: indicates if the patient has undergone thyroid surgery (Yes or No).

Query Hypothyroid: indicates if the query is related to hypothyroidism.

Query Hyperthyroid: indicates if the query is related to hyperthyroidism.

Lithium: indicates if the patient is taking lithium medication.

Goitre: indicates if the patient has a goitre (swelling of the thyroid gland).

TSH measured: indicates if the TSH level was measured.

TSH: the level of Thyroid Stimulating Hormone.

T3 measured: indicates if the T3 level was measured.

T3: the level of Total Triiodothyronine.

TT4 measured: indicates if the TT4 level was measured.

TT4: the level of Total Thyroxine.

T4U measured: indicates if the T4U level was measured.

T4U: the level of Free Thyroxine.

FTI measured: indicates if the FTI level was measured.

FTI: the level of Free Thyroid Index.

TBG measured: indicates if the TBG level was measured.

TBG: the level of Thyroid Binding Globulin.

Referral Source: the source of the referral for the patient.

Class: indicates the patient's thyroid status (hypothyroid, hyperthyroid, or euthyroid).

FTI-TSH: the difference between the FTI and TSH levels.

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


