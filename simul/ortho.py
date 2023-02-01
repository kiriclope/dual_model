import numpy as np
import matplotlib.pyplot as plt

N = 1000 
u = np.arange(N)
print(u[:10])

v = u.copy() 
rng = np.random.default_rng(None) 
rng.shuffle(v)
print(v[:10])

uv = np.dot(u,v)

w = v - uv / np.sum(u) 

print(w[:10]) 
print(np.dot(u,w)) 

theta = np.linspace(0, np.pi, N)
print(theta[:10])

phi = theta.copy()
rng.shuffle(phi)
print(phi[:10]) 

cos_between = lambda a,b : np.dot(a,b) / np.linalg.norm(a) / np.linalg.norm(b) 

theta_ortho =  (phi - np.dot(theta, phi) / np.linalg.norm(theta)**2 * theta ) 
print(theta_ortho[:10]) 
print('cos theta theta_ortho:', cos_between(theta, theta_ortho) ) 

angle = 3*np.pi/4 
theta_1 = ( phi - np.cos(angle) / np.linalg.norm(theta)**2 * theta ) 

cosine = cos_between(theta, theta_1) 
print('angle theta theta_1', np.arccos(cosine), 'angle', angle ) 

# a / ||b + c|| / ||a|| = 1 
