import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


sections = [
   {"Width": 10.5, "Velocity": 0.15, "Slope": 0.06, "Roughness": 0.04, "Length": 0.1, "Inflow": 0.02},
   {"Width": 10.6, "Velocity": 0.18, "Slope": 0.045, "Roughness": 0.038, "Length": 0.15, "Inflow": 0.019},
   {"Width": 10.7, "Velocity": 0.15, "Slope": 0.05, "Roughness": 0.039, "Length": 0.2, "Inflow": 0.021},
   {"Width": 11.0, "Velocity": 0.2, "Slope": 0.063, "Roughness": 0.041, "Length": 0.25, "Inflow": 0.018},
   {"Width": 10.9, "Velocity": 0.18, "Slope": 0.048, "Roughness": 0.037, "Length": 0.3, "Inflow": 0.023},
   {"Width": 11.0, "Velocity": 0.13, "Slope": 0.055, "Roughness": 0.036, "Length": 0.35, "Inflow": 0.022},
   {"Width": 11.1, "Velocity": 0.15, "Slope": 0.05, "Roughness": 0.035, "Length": 0.4, "Inflow": 0.024},
   {"Width": 10.8, "Velocity": 0.14, "Slope": 0.062, "Roughness": 0.034, "Length": 0.45, "Inflow": 0.025},
   {"Width": 11.2, "Velocity": 0.16, "Slope": 0.049, "Roughness": 0.033, "Length": 0.5, "Inflow": 0.027},
   {"Width": 10.6, "Velocity": 0.15, "Slope": 0.047, "Roughness": 0.032, "Length": 0.55, "Inflow": 0.026}
]


rho = 1000  
mu = 0.001  


delta_t = 0.01  
delta_x = 0.01  
T = 5.0  
N_t = int(T / delta_t)  


def calculate_reynolds_number(u, L):
   return (rho * u * L) / mu


def vary_inflow(w, t):
   seasonal_factor = np.sin(2 * np.pi * t / (365 * 24 * 3600)) 
   return w * (1 + 0.5 * seasonal_factor)


def finite_difference_method(section, F_0, beta, gamma):
   B = section["Width"]
   u = section["Velocity"]
   i = section["Slope"]
   C = section["Roughness"]
   L = section["Length"]
   w = section["Inflow"]
   Re = calculate_reynolds_number(u, L)

   section_N_x = int(L / delta_x)
   F = np.zeros((section_N_x, N_t))
   F[:, 0] = F_0  

   for t in range(1, N_t):
       w_varied = vary_inflow(w, t * delta_t)
       for x in range(1, section_N_x - 1):
           F[x, t] = F[x, t-1] + delta_t * (
               - (3/2) * C * np.sqrt(i * F[x, t-1]) * (F[x+1, t-1] - F[x-1, t-1]) / (2 * delta_x)
               + (1/Re) * (F[x+1, t-1] - 2 * F[x, t-1] + F[x-1, t-1]) / (delta_x**2)
               + B * w_varied  
           )
           F[x, t] = max(F[x, t], F_0)
       
       
       F[0, t] = (4 * F[1, t] - F[2, t]) / (3 + 2 * beta * delta_x) 
       F[-1, t] = (4 * F[-2, t] - F[-3, t]) / (3 + 2 * gamma * delta_x)

   return F


dataset = []


initial_depth = 1.0  
beta = 0.5
gamma = 0.5
total_length = sum(section["Length"] for section in sections)
N_full_x = int(total_length / delta_x)
F_full_river = np.zeros((N_full_x, N_t))

current_x = 0
for section in sections:
   F_section = finite_difference_method(section, initial_depth, beta, gamma)

   
   section_length_points = F_section.shape[0]
   F_full_river[current_x:current_x + section_length_points, :] = F_section
   
   
   for t in range(0, N_t, int(0.5 / delta_t)): 
       for x in range(0, section_length_points, 10):
           dataset.append({
               'Width': section['Width'],
               'Velocity': section['Velocity'],
               'Slope': section['Slope'],
               'Roughness': section['Roughness'],
               'Length': section['Length'],
               'Inflow': section['Inflow'],
               'Time': t * delta_t,
               'Position': current_x + x,
               'Depth': F_section[x, t]
           })
   
   current_x += section_length_points


df = pd.DataFrame(dataset)


df.to_csv('river_simulation_dataset.csv', index=False)


plt.figure(figsize=(15, 10))


times_to_plot = [0.5 * i for i in range(1, int(T / 0.5))]

for t in times_to_plot:
   subset = df[df['Time'] == t]
   x_values = subset['Position'].values
   y_values = subset['Depth'].values
   
   
   x_smooth = np.linspace(x_values.min(), x_values.max(), 300)
   spl = make_interp_spline(x_values, y_values, k=3)
   y_smooth = spl(x_smooth)
   
   plt.plot(x_smooth, y_smooth, label=f'Час {t:.2f}с', linewidth=2)
   
  
   section_positions = np.cumsum([section["Length"] for section in sections])
   for pos in section_positions[:-1]:
       x_marker = pos / delta_x
       y_marker = spl(x_marker)
       plt.plot(x_marker, y_marker, 'ro', markersize=8)

plt.xlabel('Відстань (м)', fontsize=14)
plt.ylabel('Глибина (м)', fontsize=14)
plt.title('Глибина річки з часом з плавними переходами (Метод скінченних різниць)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show() 