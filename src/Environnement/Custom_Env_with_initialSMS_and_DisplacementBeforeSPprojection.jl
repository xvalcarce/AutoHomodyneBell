
using QuantumOpticalCircuits
using Optim
using Distributions
using ReinforcementLearning
using CUDA

include("../measures_homodyne_v4.jl")


mutable struct node
    s::Function#The Gate as a symbol
    modes::Vector{Int64} #The modes on which it applies
    init_param::Float64

end


mutable struct Custom_Env <: AbstractEnv
    circuit::Vector{node} #chain of nodes(optical gates)
    N::Int64 #number of modes
    actions::Vector{node} #set of gates that the agent can choose from
    start_circuit::Vector{node}
    
    n_circuit::Int64 #max length of the circuit

    params::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    bin::Float64

    chsh_absvalue::Float64
end

function (env::Custom_Env)(action) #action is [Int]

    if length(env.circuit)==0  #we intialize the circuit with one SMS per mode
        N=env.N

        for i in 1:N
            sms=node(SMS,[i],0.8)
            push!(env.circuit,sms)
            push!(env.params,0.8)
            push!(env.lower,1e-4)
            push!(env.upper,1.8)
        end
    end
    
    a= env.actions[action[1]] #have to convert it

    push!(env.circuit,a)
    update_params!(env,a)
    
    
    if  length(env.circuit) >= env.n_circuit #we add a displacement before each heralding detector
        N=env.N
        
        for i in 1:N-2
            displ=node(D,[N+1-i],0.5)
            push!(env.circuit,displ)
            push!(env.params,0.5)
            push!(env.lower,-1.0)
            push!(env.upper,1.0)
        end
        
        #optimize_allparams!(env)
        optimize_params!(env)
    end
end

#builds circuit from list of actions
function (env::Custom_Env)(actionlist::Vector{Int64})
    for i in 1:lastindex(actionlist)
        env(actionlist[i])
    end
end


function Custom_Env(N::Int64,n_circuit::Int64) #Constructor

    start_circuit = Vector{node}(undef,0) #initial circuit
    circuit = Vector{node}(undef,0) #initial circuit also

    params = Vector{Float64}(undef,0)
    lower = Vector{Float64}(undef,0)
    upper = Vector{Float64}(undef,0)
    bin = 0.0
    
    actions = build_action_space(N)

    return Custom_Env(circuit,N,actions,start_circuit,n_circuit,params,lower,upper,bin,0.0)
end


function build_action_space(N::Int64)
    actions = Vector{node}(undef,0) #In this case we only have passive gates

    
    r=5 #rounding at r digits

    
    #here the special case with 1 copy per gate
    for n in range(1,N) 
        push!(actions,node(PS,[n],round(pi/8,digits=r)))
    end

    for i in range(1,N) 
        for j in range(i+1,N)
            push!(actions,node(BS,[i,j],round(1.5,digits=r)))
        end
    end

    return actions

end


function optimize_params!(env::Custom_Env)

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

    startingCHSH= abs(CHSH(env,initial_x))

    #if startingCHSH>1e-2 #it was 0.5 for 2 copies per gate
        m = optimize(f,lower,upper,initial_x, NelderMead(), Optim.Options(x_reltol=1e-4,f_abstol=1e-2,time_limit=500))
        optparam=[round.(x,digits=r) for x in Optim.minimizer(m)]  # the optimal values of the parameters of the nodes
    
        for i in 1:lparams
            env.params[i]=optparam[i]
        end
    #end

    env.chsh_absvalue = abs(CHSH(env,env.params))
end


function optimize_herald!(env::Custom_Env)

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
    optparam=[round.(x,digits=r) for x in Optim.minimizer(m)]  # the optimal values of the parameters of the nodes
    
    for i in 1:lparams
        env.params[i]=optparam[i]
    end

    env.chsh_absvalue = abs(CHSH(env,env.params))
    return heraldprob(env,env.params)

end


function optimize_paramslong!(env::Custom_Env)

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

    startingCHSH= abs(CHSH(env,initial_x))

    #if startingCHSH>1e-2 #it was 0.5 for 2 copies per gate
        m = optimize(f,lower,upper,initial_x, NelderMead(), Optim.Options(x_reltol=1e-4,f_abstol=1e-2,time_limit=5000))
        optparam=[round.(x,digits=r) for x in Optim.minimizer(m)]  # the optimal values of the parameters of the nodes
    
        for i in 1:lparams
            env.params[i]=optparam[i]
        end
    #end

    env.chsh_absvalue = abs(CHSH(env,env.params))
end


#=
#this function optimizes simultaneously over the gates parameters and over the binning width x
function optimize_allparams!(env::Custom_Env)

    
    lower = [env.lower;-1.2]
    upper = [env.upper; 1.2]
    initial_x = [env.params; env.bin]
    lparams = length(initial_x)
    middle=Vector{Float64}(undef,0)
    hfwidth=Vector{Float64}(undef,0)
    r = 5 #rounding parameter
    
    for i in 1:lparams
        push!(middle, (lower[i]+upper[i])/2)
        push!(hfwidth, (upper[i]-lower[i])/2)
    end

    function f(x)  #penalty for evaluating function outside of range
        for i in 1:lparams
            if abs(x[i]-middle[i])>hfwidth[i]
                return exp(abs(x[i]-middle[i]))
            end
        end
        return -abs(CHSHbin(env,x))
    end

    startingCHSH= abs(CHSHbin(env,initial_x))

    if startingCHSH>1e-2 #it was 0.5 for 2 copies per gate
        m = optimize(f,lower,upper,initial_x, NelderMead(), Optim.Options(x_reltol=1e-4,f_abstol=1e-2,time_limit=500))
        optparam=[round.(x,digits=r) for x in Optim.minimizer(m)]  # the optimal values of the parameters of the nodes
    
        for i in 1:lparams-1
            env.params[i]=optparam[i]
        end
        env.bin=optparam[lparams]
    end

    paramsbin=[env.params;env.bin]

    env.chsh_absvalue = abs(CHSHbin(env,paramsbin))
end
=#


function update_params!(env::Custom_Env,action::node) #add the parameter and upper and lower bound
    
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



function apply_hook!(env::Custom_Env,hook,i::Int64)

    env.circuit = hook.circuit[i]
    env.params = hook.params[i]
    env.bin = hook.bin[i]

    return CHSH(env,env.params)
end

function CHSH(env::Custom_Env,params::Vector{Float64})
    
    state = state_inner(env,params)

    if state[2]==0
        return 0.0
    else
        return CHSH(state[1])
    end
end

#=
function CHSHbin(env::Custom_Env,paramsbin::Vector{Float64})
    
    params=paramsbin[1:lastindex(paramsbin)-1]
    bin=last(paramsbin)
    state = state_inner(env,params)

    if state[2]==0
        return 0.0
    else
        return CHSHbin(state[1],bin)
    end
end

function apply_hook!(env::Custom_Env,hook,i::Int64)

    env.circuit = hook.circuit[i]
    env.params = hook.params[i]
    env.bin = hook.bin[i]
    paramsbin=[env.params;env.bin]

    return CHSHbin(env,paramsbin)
end
=#


function state_inner(env::Custom_Env) 
    return state_inner(env,env.params)
end


function heraldprob(env::Custom_Env,params::Vector{Float64})
    state = state_inner(env,params)

    if state[2]==0
        return 0.0
    else
        return 1/state[1].norm
    end
end



# We redefine the function state_inner (which computes the state produced by the circuit at the end of an episode) by removing the Heralding and instead checking at each step if the probability of click is above tolerance. If so, the function returns the state of the curcuit and a 1, otherwise it returns a dummy state and a zero. The bit then indicates to CHSH(env::Custom_Env,params::Vector{Float64}) and to RLBase.state(env::Custom_Env) whether the returned state is to be trusted. 
function state_inner(env::Custom_Env,params::Vector{Float64})
    N = env.N 
    circuit = env.circuit

    W = PseudoGaussianState(N+N-2) # we add the N-2 modes initialized in vacuum to implement the photon subtraction
    L = size(circuit)[1] #number of actual gates

    dummy=PseudoGaussianState(2) #dummy state returned in case the heralding probability is too low

    for l in range(1,L) 
        W = circuit[l].s(params[l]...)(circuit[l].modes...)(W) 
    end
    
    #Here we perform single-photon projection on the modes: N, N-1, ..., 3
    #first we split one photon off from the N-2 modes
    for i in 1:N-2 
        W |> BS(0.1)(N+1-i,N+N-2 +1-i)
    end

    #here we herald a no-click on the modes from which we subtracted a photon
    for i in 1:N-2
        if PhotonDetector(N+1-i,η=1.0,tol=1e-10)(W)>1e-10 && 1-PhotonDetector(N+1-i,η=1.0,tol=1e-10)(W) > 1e-10 
            Heralding(N+1-i,η=1.0,tol=1e-10,onclick=false)(W)
        else
            return dummy, 0
        end
    end

    #then we herald a click in the last N-2 modes, where we added a photon with the BS
    for i in 1:N-2
        if PhotonDetector(N+1-i,η=1.0,tol=1e-10)(W) > 1e-10 
            Heralding(N+1-i,η=1.0,tol=1e-10,onclick=true)(W)
        else
            return dummy, 0
        end
    end
    

    return W, 1
end

#this function computes the gaussian state produced by the circuit stored in env
function state_innergauss(env::Custom_Env)
    N = env.N 
    circuit = env.circuit
    params=env.params

    W = PseudoGaussianState(N+N-2) 
    L = size(circuit)[1] #number of gates


    for l in range(1,L) 
        W = circuit[l].s(params[l]...)(circuit[l].modes...)(W) 
    end
    
    return W
end



RLBase.action_space(env::Custom_Env) = Base.OneTo(length(env.actions)) # env.actions 


function RLBase.state(env::Custom_Env) #This is the state produced by the circuit, as passed to the learning agent. We can see that we pass the weight of the Gaussian state in the PseudoGaussian mixture, and then we pass the elements of the upper triangle of its CM (since it is symmetric, we don't pass the whole matrix). Note that this needs to be modified if we want to include displaced states!!

    W=state_inner(env) #the state produced by the circuit
    input = Vector{Float64}(undef,0) #input given to learning agent
    N = env.N
    M = 2^(N-2)   #number of gaussian states in mixture; this is 2^(N-2) with N-2 detectors Heralding a click

    if W[2] == 0 #in case that heralding doesn't work, we pass a vacuum state in each mode
        
        vac=GaussianState(2)
        for i in 1:M 
            push!(input,1/M)
            append!(input,vec_symmetric(vac.σ))  #we pass a vacuum state in each mode
        end
        return input

    else

        state=W[1]

        for i in range(1,M) 
            push!(input,state.prob[i])
            append!(input,vec_symmetric(state.states[i].σ))
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

RLBase.state_space(env::Custom_Env) = Vector{Float64}(undef,2^(N-2)*n_0)
RLBase.is_terminated(env::Custom_Env) = ( size(env.circuit)[1] == env.n_circuit+env.N-2) #condition to terminate the episode; we add N-2 to the length of the circuit because we have the displacement operators


function RLBase.reset!(env::Custom_Env) #Reset everything after one episode

    env.params = Vector{Float64}(undef,0)
    env.lower = Vector{Float64}(undef,0)
    env.upper = Vector{Float64}(undef,0)
    env.circuit = copy(env.start_circuit)
    env.chsh_absvalue = 0 
end

#this is the reward function that works best according to our tests; the linear part is in [-1,-1/2], while the exponential is in [0,1]
function RLBase.reward(env::Custom_Env)
    if length(env.circuit)< env.n_circuit
        return 0.
    else
        if env.chsh_absvalue<2
            return env.chsh_absvalue/4. -1
        else 
            return exp(10*log(2)*(env.chsh_absvalue-2))-1  #this function is equal to 0 for chsh=2 and to 1 when chsh=2.1
        end
    end
end


RLBase.NumAgentStyle(::Custom_Env) = SINGLE_AGENT
RLBase.DynamicStyle(::Custom_Env) = SEQUENTIAL
RLBase.ActionStyle(::Custom_Env) = MINIMAL_ACTION_SET
RLBase.InformationStyle(::Custom_Env) = PERFECT_INFORMATION
RLBase.StateStyle(::Custom_Env) = Observation{Int}()
RLBase.RewardStyle(::Custom_Env) = TERMINAL_REWARD
RLBase.UtilityStyle(::Custom_Env) = GENERAL_SUM
RLBase.ChanceStyle(::Custom_Env) = DETERMINISTIC

