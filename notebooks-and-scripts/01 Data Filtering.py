# -*- coding: utf-8 -*-
"""
00 DATA FILTERING

Data Cleaning for ANES Cumulative Time Series Survey

Input: anes_cdf_raw.csv
    The input file is a CSV containing compelte data on all survey 
    respondents from 1948-2012. 
    
Output: anes_cdf_abridged.csv
    The output file is a CSV containing only data on survey respondents from
    presidential election years 2000, 2004, 2008, and 2012. Certain features
    are also removed from the data, as documented below.
"""

import pandas as pd

############################
### RESPONDENT FILTERING ###
############################

# Read in full ANES CDF survey
full_df = pd.read_csv('../data/anes_cdf_raw.csv', encoding = 'utf-8')
    # full_df contains 55674 respondents and 953 features

# Keep responses from presidential election years 2000 and later
df = full_df[full_df.VCF0004 >= 2000]
    # removes 42908 respondents, df shape: (12766, 953)
df = df[df.VCF0004 != 2002]
    # removes 1511 respondents, df shape: (11255, 953)

# Drop all respondents for which no post-election interview data is present
df = df[df.VCF0013.str.contains('1')]
df = df.drop('VCF0013', axis = 1) 
    # removes 1022 respondents and 1 feature, df shape: (10233, 952)

# Drop all respondents with abbreviated pre-election interviews
df = df[df.VCF0015a.str.startswith('0')]
df = df.drop('VCF0015a', axis = 1)
    # removes 836 repsondents and 1 feature, df shape: (9397, 951)
    
# Drop all respondents with no data on whether they voted
df = df[~df.VCF0702.str.startswith('0')]
    # removes 23 respondents, df shape: (9374, 951)

#########################
### FEATURE FILTERING ###
#########################

# Because the survey is not a simple random sample, each respondent is ...
# assigned a weight for the purpose of inferring properties of the ...
# general population. Because Type 0, Type 1, and Type 2 weights are
# are the same for respondents interviewed in 2000 and later, we keep
# only one set of full sample weights.

columns = ['VCF0009x','VCF0009y','VCF0009z', 'VCF0010x','VCF0010y',
                'VCF0010z','VCF0011x','VCF0011y']
df = df.drop(columns, axis = 1)
    # removes 8 features, df shape: (9374, 943)
    
# The following features are dropped due to lack of relevance or
# redundancy.
# Unnamed: 0 - Interviewee ID number
# Version - Version release number
# VCF0006 - Study respondent number
# VCF0006a - Unique respondent number
# VCF0012 - Form/interview type of paper questionnaire
# VCF0012a - CAI Question selection (pre)
# VCF0012b - CAI Question selection (post)
# VCF0014 Pre-election interview data present
# VCF0015b - Abbreviated interview (post)
# VCF0016 - Cross case
# VCF0017 - Interview type
# VCF0018a - Language of interview (pre)
# VCF0018b - Langage of interview (post)
# VCF0019 - Relationship to head of household
# VCF0070a - Interviewer gender (pre)
# VCF0070b - Interviewer gender (post)
# VCF0071a - Interviewer race (pre)
# VCF0071b - Interviewer race (post)
# VCF0071c - Interviewer race, 2-category (pre)
# VCF0071d - Interviewer race, 2-category (post)
# VCF0072a - Interviewer ethnicity (pre)
# VCF0072b - Interviewer ethnicity (post)
# VCF0102 - Age group
# VCF0103 - Cohort
# VCF0106 - Race summary, 3-category
# VCF0109 - Ethnicity (too any categories)

columns = ['Unnamed: 0', 'Version','VCF0006','VCF0006a','VCF0012','VCF0012a',
           'VCF0012b', 'VCF0015b','VCF0016','VCF0019','VCF0070a','VCF0070b',
           'VCF0071a','VCF0071b','VCF0071c','VCF0071d','VCF0072a','VCF0072b',
           'VCF0106','VCF0109','VCF0102','VCF0103','VCF0014','VCF0018a',
           'VCF0018b','VCF0017']
df = df.drop(columns, axis = 1)
    # removes 22 features, df shape: (9397, 917)

# Drop all features with substantial amounts of missing data
df = df.loc[:, df.notnull().sum(axis = 0) >= 6000]
    # removes 654 features, df shape: (9374, 263)


df.to_csv('../data/anes_cdf_abridged.csv')