using QuantumOpticalCircuits
using Optim
using Distributions
using ReinforcementLearning
using CUDA

include("../measures_homodyne.jl")

mutable struct action
    s::Function #The Gate as a symbol
    modes::Vector{Int64} #The modes on which it applies
    init_param::Float64
end

mutable struct Env <: AbstractEnv
    circuit::Vector{action} #chain of action(optical gates)
    N::Int64 #number of modes
    actions::Vector{action} #set of gates that the agent can choose from
    start_circuit::Vector{action} #initial circuit
    n_circuit::Int64 #max length of the circuit
    params::Vector{Float64} # circuit parameters
	lower::Vector{Float64} #lower bound on circuit parameters (for optimisation)
    upper::Vector{Float64} #upper bound
    chsh_absvalue::Float64 # current chsh value
end

function (env::Env)(action::Int)
	""" Apply a gate on the circuit """
    a = env.actions[action[1]] #have to convert it
    push!(env.circuit,a)
    update_params!(env,a)
    if  length(env.circuit) >= env.n_circuit #we add a displacement before each heralding detector
        optimize_params!(env)
    end
end

#builds circuit from list of actions
function (env::Env)(actionlist::Vector{Int64})
    for i in 1:lastindex(actionlist)
        env(actionlist[i])
    end
end

function optimize_params!(env::Env;time_limit=500,x_reltol=1e-4,f_abstol=1e-2)
    lower = env.lower
    upper = env.upper
    initial_x = env.params
    lparams = length(env.params)
    middle=Vector{Float64}(undef,0)
    hfwidth=Vector{Float64}(undef,0)
    r = 5 #rounding parameter
    for i in 1:length(initial_x)
        push!(middle, (lower[i]+upper[i])/2)
        push!(hfwidth, (upper[i]-lower[i])/2)
    end
    function f(x)  #penalty for evaluating function outside of range
        for i in 1:lparams
            if abs(x[i]-middle[i])>hfwidth[i]
                return exp(abs(x[i]-middle[i]))
            end
        end
        return -abs(CHSH(env,x))
    end
    startingCHSH = abs(CHSH(env,initial_x))
    if startingCHSH > 1e-2 
        m = optimize(f,lower,upper,initial_x, NelderMead(), Optim.Options(x_reltol=x_reltol,f_abstol=f_abstol,time_limit=time_limit))
        optparam=[round.(x,digits=r) for x in Optim.minimizer(m)]  # the optimal values of the parameters of the actions
        for i in 1:lparams
            env.params[i]=optparam[i]
        end
    end
    env.chsh_absvalue = abs(CHSH(env,env.params))
end

optimize_paramslong!(env::Env) = optimize_params!(env,time_limit=900, x_reltol=1e-6,f_abstol=1e-4)


function optimize_herald!(env::Env)
    lower = env.lower
    upper = env.upper
    initial_x = env.params
    lparams = length(env.params)
    middle=Vector{Float64}(undef,0)
    hfwidth=Vector{Float64}(undef,0)
    r = 5 #rounding parameter
    for i in 1:length(initial_x)
        push!(middle, (lower[i]+upper[i])/2)
        push!(hfwidth, (upper[i]-lower[i])/2)
    end
    function f(x)  #penalty for evaluating function outside of range and if the violation drops below 2.068
        for i in 1:lparams
            if abs(x[i]-middle[i])>hfwidth[i]
                return exp(abs(x[i]-middle[i]))
            end
        end
        if abs(CHSH(env,x))< 2.0675
            return exp(abs(2.0675-CHSH(env,x)))
        end
        return -heraldprob(env,x)
    end
    m = optimize(f,lower,upper,initial_x, NelderMead(), Optim.Options(x_reltol=1e-4,time_limit=500))
    optparam=[round.(x,digits=r) for x in Optim.minimizer(m)]  # the optimal values of the parameters of the actions
    for i in 1:lparams
        env.params[i]=optparam[i]
    end
    env.chsh_absvalue = abs(CHSH(env,env.params))
    return heraldprob(env,env.params)
end


function update_params!(env::Env,action::action) #add the parameter and upper and lower bound
    S_min = 1e-4 #minimum squeezing
    S_max = 2 #maximum squeezing
    if action.s == PS 
        push!(env.params,action.init_param)
        push!(env.lower,1e-8)
        push!(env.upper,pi)
    elseif action.s == BS
        push!(env.params,action.init_param)
        push!(env.lower,1e-4)
        push!(env.upper,pi)
    elseif action.s == SMS
        push!(env.params,action.init_param)
        push!(env.lower,S_min)
        push!(env.upper,S_max)
    elseif action.s == TMS
        push!(env.params,action.init_param)
        push!(env.lower,S_min)
        push!(env.upper,S_max)
    end
end



function apply_hook!(env::Env,hook,i::Int64)
    env.circuit = hook.circuit[i]
    env.params = hook.params[i]
    return CHSH(env,env.params)
end


function CHSH(env::Env,params::Vector{Float64})
    state = state_inner(env,params)
    if state[2]==0
        return 0.0
    else
        return CHSH(state[1])
    end
end

state_inner(env::Env) = state_inner(env,env.params)

function heraldprob(env::Env,params::Vector{Float64})
    state = state_inner(env,params)
    if state[2]==0
        return 0.0
    else
        return 1/state[1].norm
    end
end


# We redefine the function state_inner (which computes the state produced by the circuit at the end of an episode) by removing the Heralding and instead checking at each step if the probability of click is above tolerance. If so, the function returns the state of the curcuit and a 1, otherwise it returns a dummy state and a zero. The bit then indicates to CHSH(env::Env,params::Vector{Float64}) and to RLBase.state(env::Env) whether the returned state is to be trusted. 
function state_inner(env::Env,params::Vector{Float64})
    N = env.N 
    circuit = env.circuit
    #W = PseudoGaussianState(N+N-2) # for single-photon projection
    W = PseudoGaussianState(N) # for normal heralding
    L = size(circuit)[1] #number of actual gates
    dummy = PseudoGaussianState(2) #dummy state returned in case the heralding probability is too low
    for l in range(1,L) 
        W = circuit[l].s(params[l]...)(circuit[l].modes...)(W) 
    end
    Gaussify!(W)  # to avoid errors due to non positive-definite matrices
    # NORMAL HERALDING
    #Herald the modes: N, N-1, ..., 3 (we keep only the first two modes)
    for i in 1:N-2
        if PhotonDetector(N+1-i,η=1.0,tol=1e-10)(W) > 1e-10 
            Heralding(N+1-i,η=1.0,tol=1e-10,onclick=true)(W)
        else
            return dummy, 0
        end 
    end
    return W, 1
end

RLBase.action_space(env::Env) = Base.OneTo(length(env.actions)) # env.actions 


function RLBase.state(env::Env) #This is the state produced by the circuit, as passed to the learning agent. We can see that we pass the weight of the Gaussian state in the PseudoGaussian mixture, and then we pass the elements of the upper triangle of its CM (since it is symmetric, we don't pass the whole matrix). Note that this needs to be modified if we want to include displaced states!!
    W = state_inner(env) #the state produced by the circuit
    input = Vector{Float64}(undef,0) #input given to learning agent
    N = env.N
    M = 2^(N-2)  #number of gaussian states in mixture
    vac=GaussianState(2)
    if W[2] == 0 #in case that heralding doesn't work, we pass a vacuum state in each mode
        for i in 1:M 
            push!(input,1/M)
            append!(input,vec_symmetric(vac.σ))  #we pass a vacuum state in each mode
        end
        return input
    else
        state=W[1]
        for i in 1:lastindex(state.prob)
            push!(input,state.prob[i])
           append!(input,vec_symmetric(state.states[i].σ))
        end
        if lastindex(state.prob)<M
            for j in 1:M-lastindex(state.prob)
                push!(input,0.0)
                append!(input,vec_symmetric(vac.σ))
            end
        end
        return input
    end
end

function vec_symmetric(M::Matrix{Float64}) #Vectorize a symmetric matrix
    N = size(M)[1]
    v=Vector{Float64}(undef,0)
    for i in range(1,N)
        for j in range(1,i)
            push!(v,M[i,j])
        end
    end
    return v
end

RLBase.state_space(env::Env) = Vector{Float64}(undef,2^(N-2)*n_0)
RLBase.is_terminated(env::Env) = ( size(env.circuit)[1] == env.n_circuit) #condition to terminate the episode;


function RLBase.reset!(env::Env) #Reset everything after one episode
    env.params = Vector{Float64}(undef,0)
    env.lower = Vector{Float64}(undef,0)
    env.upper = Vector{Float64}(undef,0)
    env.circuit = copy(env.start_circuit)
    env.chsh_absvalue = 0 
end

function RLBase.reward(env::Env)
    if env.chsh_absvalue<1e-10  #this happens if the Heralding probability is below tolerance (see CHSH and state_inner) or if the norm of the state is instable due to numerical errors
        return -100
    elseif env.chsh_absvalue<=2
        return 10*env.chsh_absvalue
    else
        return 20*exp(2*env.chsh_absvalue)
    end
end

RLBase.NumAgentStyle(::Env) = SINGLE_AGENT
RLBase.DynamicStyle(::Env) = SEQUENTIAL
RLBase.ActionStyle(::Env) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::Env) = PERFECT_INFORMATION
RLBase.StateStyle(::Env) = Observation{Int}()
RLBase.RewardStyle(::Env) = TERMINAL_REWARD
RLBase.UtilityStyle(::Env) = GENERAL_SUM
RLBase.ChanceStyle(::Env) = DETERMINISTIC
