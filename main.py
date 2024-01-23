from flexprofilegeneratorbuilder import FlexProfileGeneratorBuilder
import time

if __name__ == "__main__":
    # house type is: 'vrijst', '2_1kap', 'hoek', 'tussen', 'maisonette', 'galerij', 'portiek', or 'overig'
    # building year is: '1975 - 1991', '1992 - 2005', '2006 - 2012', or '> 2012'
    # residents type is 'one_person', 'two_person', or 'family'

    house_type = '2_1kap'
    house_year = '> 2012'
    residents_type = 'one_person'

    house_params = {'house_type': house_type,
                    'house_year': house_year,
                    'residents_type': residents_type}

    start_congestion_month = 1  # January
    start_congestion_day = 15  # 15th day
    start_congestion_ptu = 48  # at 12:00
    end_congestion_ptu = 72  # 18:00, this value could also be e.g. 10 (2:30), on the next day

    # Take 6 hours of window before the congestion and 18 - congestion duration after the congestion
    window_before_congestion_ptu = 6 * 4
    congestion_length = end_congestion_ptu - start_congestion_ptu
    congestion_length = congestion_length if end_congestion_ptu >= start_congestion_ptu else congestion_length + 96
    window_after_congestion_ptu = 18 * 4 - congestion_length

    congestion_params = {'start_congestion_month': start_congestion_month,
                         'start_congestion_day': start_congestion_day,
                         'start_congestion_ptu': start_congestion_ptu,
                         'end_congestion_ptu': end_congestion_ptu,
                         'window_before_congestion_ptu': window_before_congestion_ptu,
                         'window_after_congestion_ptu': window_after_congestion_ptu}

    # Build the generator
    flex_profile_generator_builder = FlexProfileGeneratorBuilder()
    flex_profile_generator = flex_profile_generator_builder.build_empty()
    flex_profile_generator = flex_profile_generator_builder.build_physical_attributes(house_params)
    flex_profile_generator = flex_profile_generator_builder.build_congestion_attributes(congestion_params)

    # Generate profiles
    flex_profiles = flex_profile_generator.generate(n_profiles=50)

    # Store resulting profiles
    file_name = 'flex_profiles+' + house_type + '+' + house_year.replace(" ", "").replace(">", "") \
                + '+' + residents_type + '.csv'
    flex_profiles.to_csv(file_name)

    '''
    Note that Builder class can be abused to dynamically construct a FlexProfileGenerator in a loop.
    this way you can loop over house types or congestion chracteristics
    An example:
    
    flex_profile_generator_builder = FlexProfileGeneratorBuilder()
    flex_profile_generator = flex_profile_generator_builder.build_empty()
    flex_profile_generator = flex_profile_generator_builder.build_congestion_attributes(congestion_params)
    iterator_house_params = [....]
    for house_params in iterator_house_params:
        flex_profile_generator = flex_profile_generator_builder.build_physical_attributes(house_params)
        flex_profiles = flex_profile_generator.generate(n_profiles=50)
        house_type = house_params['house_type']
        house_year = house_params['house_year']
        residents_type = house_params['residents_type']
        file_name = 'flex_profiles+' + house_type + '+' + house_year.replace(" ", "").replace(">", "") \
                + '+' + residents_type + '.csv'
        flex_profiles.to_csv(file_name)
    '''