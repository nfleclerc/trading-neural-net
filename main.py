import training
import model as m
import time

ticker = 'AAPL'
start_year = 2015
end_year = 2020
future_prediction = 3
time_frame = 60
validation_split = 0.2

tx, ty, vx, vy = training.create_data(ticker=ticker, start_year=start_year, end_year=end_year,
                                      validation_split=validation_split,
                                      future_prediction=future_prediction, time_frame=time_frame)

model = m.create_model(f'{ticker}-{start_year}-{end_year}-{time_frame}-{future_prediction}-{time.time()}')