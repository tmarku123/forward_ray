include("functions.jl");

filename = "seabed.png";
domain   = initialise_domain(filename;plot=false);

dvdx          = 0;
dvdz          = 0.01;
base_velocity = 1.4;        

velocity_grads = VelocityGradients(dvdx,dvdz,base_velocity);
fields         = initialise_fields(domain,velocity_grads;plot=false);

t_max       = 250;
no_rays     = 10;
angle_range = [0 , 360];
params      = initialise_parameters(t_max,no_rays,angle_range,fields);

x_0       = domain.x_dims/2;  
z_0       = 400;
position  = [x_0 , z_0];
amp_0     = 1;
theta     = params.angles[1];
u_initial = interpolate_field(fields.slowness_field,x_0,z_0);

source_rays = Vector{Ray}(undef, size(params.angles,1));
source_data = Vector{RayData}(undef, size(params.angles,1))
no_samples  = 1000

for (i,theta) in enumerate(params.angles)
    ICs      = InitialConditions(position,amp_0,theta,u_initial)
    ray_data = initialise_ray_data(params,no_samples)
    source_rays[i],source_data[i] = calculate_ray(ICs,domain,params,fields,ray_data)
end

plot_ray(domain,source_data)
