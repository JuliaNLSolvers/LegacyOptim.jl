module LegacyOptim

using Optim

import Optim: Optimizer, perform_linesearch!, promote_objtype, value_gradient!,
              gradient, value, initial_state, trace!, update!, update_state!,
              update_g!, 
              @add_linesearch_fields, @initial_linesearch,
              LineSearches

export AcceleratedGradientDescent, MomentumGradientDescent

include("multivariate/solvers/accelerated_gradient_descent.jl")
include("multivariate/solvers/momentum_gradient_descent.jl")

# I think this should be trait based instead of FirstOrderSolvers etc being
# a Union of Optim types (makes it hard to expand in other packages!)
promote_objtype(method::Union{AcceleratedGradientDescent, MomentumGradientDescent}, initial_x, obj_args...) = OnceDifferentiable(obj_args..., initial_x)
promote_objtype(method::Union{AcceleratedGradientDescent, MomentumGradientDescent}, initial_x, od::OnceDifferentiable) = od
promote_objtype(method::Union{AcceleratedGradientDescent, MomentumGradientDescent}, initial_x, td::TwiceDifferentiable) = td
function update_g!(d, state, method::M) where M<:Union{AcceleratedGradientDescent, MomentumGradientDescent}
    # Update the function value and gradient
    value_gradient!(d, state.x)
end

function trace!(tr, d, state, iteration, method::Union{AcceleratedGradientDescent, MomentumGradientDescent}, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["Current step size"] = state.alpha
    end
    g_norm = vecnorm(gradient(d), Inf)
    update!(tr,
            iteration,
            value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end


end # module
