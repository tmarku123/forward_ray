include("functions.jl")

filename = "seabed.png"
domain = initialise_domain(filename)

x_grad = 0
z_grad = 1
base_velocity = 10

velocity_grads =    Velocity_Gradients(x_grad,z_grad,base_velocity)
domain_fields =     initialise_fields(domain,velocity_grads)

ds = 0.5
s_max = 1500
x_init = 750
z_init = 450
amp_init = 1
no_rays = 360
theta_init = collect(range(0,(360-360/no_rays),no_rays)) 

rays = Vector{Ray}()
for theta in theta_init
    parameters = (
        x_init,
        z_init,
        amp_init,
        theta,
        ds,
        s_max
    )
    init_conditions = Initial_Conditions(parameters...)
    ray = trace_ray(domain,domain_fields,init_conditions)
    push!(rays, ray)
end

plot_ray(domain,rays)

