import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def prepare_data_and_forecast(df, forecast_periods=12, freq='M'):
    """
    Cleans the input DataFrame, adds holidays, trains a Prophet model,
    and generates a sales forecast.

    Args:
        df (pd.DataFrame): The input DataFrame with 'ds' and 'y' columns.
        forecast_periods (int): The number of periods to forecast into the future.
        freq (str): The frequency of the forecast periods (e.g., 'D', 'W', 'M', 'Y').

    Returns:
        pd.DataFrame: A DataFrame containing the forecast results.
    """
    # --- Data Cleaning ---
    df = df.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    # Handle missing values by dropping them
    df_prophet = df

    df_prophet = df_prophet.dropna(subset=['ds', 'y'])
    holidays = pd.DataFrame({
        'holiday': 'Christmas',
        'ds': pd.to_datetime(['2022-12-25', '2023-12-25', '2024-12-25']),
        'lower_window': -1,
        'upper_window': 0,
    })
    
    # You can add more holidays like New Year's Day, etc.
    holidays = pd.concat([holidays, pd.DataFrame({
        'holiday': 'New Year\'s Day',
        'ds': pd.to_datetime(['2022-01-01', '2023-01-01', '2024-01-01']),
        'lower_window': -1,
        'upper_window': 0,
    })])

    # --- 2. Initialize and Train the Model ---
    # Create an instance of the Prophet model and add the holidays.
    m = Prophet(holidays=holidays, interval_width=0.95)

    print("Fitting the Prophet model...")
    m.fit(df_prophet)
    print("Model fitting complete.")

    # --- 3. Create Future Forecasts ---
    # Create a DataFrame with future dates for forecasting.
    future = m.make_future_dataframe(periods=forecast_periods, freq=freq)
    
    # Use the model to predict sales for the future dates.
    forecast = m.predict(future)

    # --- 4. Visualize and Return Results ---
    # Plot the forecast to see the results.
    fig1 = m.plot(forecast)
    plt.title("Sales Forecast (with Holidays)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()
    
    # Save the relevant forecast data.
    forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_export.columns = ['Date', 'Forecasted_Sales', 'Lower_Bound', 'Upper_Bound']
    forecast_export.to_csv('sales_forecast_for_power_bi.csv', index=False)
    
    print("\nForecast data saved to 'sales_forecast_for_power_bi.csv'.")
    print("This file is ready for import into Power BI.")
    
    return forecast

# --- Main Script Execution ---
if __name__ == "__main__":
    try:
        # Load your actual dataset here.
        # Example: sales_df = pd.read_csv('Superstore_Sales_Dataset.csv')

        # Placeholder for sample data to make the script runnable.
        sample_data = {
            'Order Date': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05',
                                         '2022-12-24', '2022-12-25', '2022-12-26', '2023-01-01', '2023-01-02']),
            'Sales': [100.5, 120.0, 150.2, 110.8, 95.5, 250.0, 350.0, 200.0, 180.0, 150.0]
        }
        sales_df = pd.DataFrame(sample_data)

        # Call the function to prepare data and generate the forecast.
        final_forecast = prepare_data_and_forecast(sales_df)

    except FileNotFoundError:
        print("Error: The data file was not found. Please check the file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
