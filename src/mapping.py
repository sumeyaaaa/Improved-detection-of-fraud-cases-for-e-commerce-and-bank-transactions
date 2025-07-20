def map_ip_to_country(fraud_df, ip_df):
    def get_country(ip):
        # Step 1: Filter ip_df to find matching row where IP is in range
        match = ip_df[(ip_df['lower_bound_ip_address'] <= ip) & 
                      (ip_df['upper_bound_ip_address'] >= ip)]
        
        # Step 2: Return the country if a match is found, else 'Unknown'
        if not match.empty:
            return match.iloc[0]['country']  # First matching row
        else:
            return 'Unknown'
        
    # Step 3: Apply this to every IP in the fraud data
    fraud_df['country'] = fraud_df['ip_int'].apply(get_country)
    return fraud_df
