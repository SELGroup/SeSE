# For long-form generation experiments, we evaluate on two challenging datasets featuring real-world entities from Wikipedia: FActScore and PopQA . 
# To ensure data quality, we exclude entities with multiple Wikipedia entries or those with pages shorter than 2,000 tokens. 
# For each dataset, we randomly sample 100 entities and generate a set of claims for each entity. This process yields over 1800 labeled claims (true or false) 
# on average for each model-dataset combination. 
# The long-form data will be made open source after the article is accepted.
