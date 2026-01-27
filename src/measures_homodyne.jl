using LinearAlgebra
using QuadGK
using SpecialFunctions

function PartialInt(c,sqd,x0,x1,y0,y1)
    f(y)=exp(-y^2)*(erf((x1-c*y)/sqd) - erf((x0-c*y)/sqd))
    return quadgk(f, y0, y1, rtol=1e-12, atol=0)[1]/sqrt(pi)
end

# A new approach to calculate the expectation values, with algebraic simplifications but without the rescaling with the largest eigenvalue (is it faster than the previous E in measures_homodyne_v3). The function takes as inputs: the binning of Alice (a boolean vector where 0 means outcome=+1 and 1 means -1; a vector containing the extremes of integration for each bin), the binning of Bob, and the PseudoGaussian state shared by Alice and Bob--after the local operations related to the choice of settings.
# this function is improved compared to the previous one since it does not assume that the integral over all alphaXbeta is normalized to 150
function E(a::BitVector,b::BitVector,alpha::Vector,beta::Vector,rho::PseudoGaussianState)
    summand=0
    w=rho.prob
    gaussrho=rho.states
    
    for k in 1:length(w)
        sigma=gaussrho[k].σ[[1,3],[1,3]]  #2*gaussrho[k].σ[[1,3],[1,3]] if we want to rescale the quadratures to the definitions used in my notes 
        mu=gaussrho[k].d[[1,3],1]   #sqrt(2)*gaussrho[k].d[[1,3],1] if we want to rescale the quadratures to the definitions used in my notes
        sc=sigma[1,2]
        sb=sigma[2,2]
        sqd=sqrt(det(sigma))
        
        summand0=0
        summand1=0

        for r in 1:lastindex(a)
            for s in 1:lastindex(b)
                x0=sqrt(sb)*(alpha[r]-mu[1])/sqrt(2)
                x1=sqrt(sb)*(alpha[r+1]-mu[1])/sqrt(2)
                y0=(beta[s]-mu[2])/sqrt(2*sb)
                y1=(beta[s+1]-mu[2])/sqrt(2*sb)
                if a[r] - b[s] == 0
                    summand0 += PartialInt(sc,sqd,x0,x1,y0,y1)/2
                else
                    summand1 += PartialInt(sc,sqd,x0,x1,y0,y1)/2
                end
            end
        end
        summand +=  w[k]*(summand0-summand1)
    end
    return summand*rho.norm
end



function CHSH(rho::PseudoGaussianState)
    if abs(rho.norm) > 1e12 # if the norm is too big, the numerical imprecision of the numerical integrals (tolerance is 1e-12) spoils the calculation of E
        return 0
    else
        #we create 4 copies of the circuit state, one for each setting combination. Then we process each copy of the state according to the settings chosen by Alice and Bob.
        rho00=copy(rho)
        rho01=copy(rho)
        rho10=copy(rho)
        rho11=copy(rho)

        # setting (A_0,B_0) -- phase shifts: (0,-π/4)
        rho00 = rho00 |> PS(0.0)(1) |> PS(-pi/4)(2)

        # setting (A_0,B_1) -- phase shifts: (0,π/4)
        rho01 = rho01 |> PS(0.0)(1) |> PS(pi/4)(2)

        # setting (A_1,B_0) -- phase shifts: (π/2,-π/4)
        rho10 = rho10 |> PS(pi/2)(1) |> PS(-pi/4)(2) 

        # setting (A_1,B_1) -- phase shifts: (π/2,π/4)
        rho11 = rho11 |> PS(pi/2)(1) |> PS(pi/4)(2) 
        
        # the sign binning:
        a=BitArray([1,0])
        b=a
        alpha=[-Inf,0,Inf]  #increased extremes to capture Wigner functions of highly squeezed states
        beta=alpha
         return E(a,b,alpha,beta,rho00) + E(a,b,alpha,beta,rho01) + E(a,b,alpha,beta,rho10) - E(a,b,alpha,beta,rho11)
    end
end


function CHSHbin(rho::PseudoGaussianState,x::Float64,y::Float64)
    
    
    if rho.norm > 1e12 # if the norm is too big, the numerical imprecision of the numerical integrals (tolerance is 1e-12) spoils the calculation of E
        return 0
    else
        #we create 4 copies of the circuit state, one for each setting combination. Then we process each copy of the state according to the settings chosen by Alice and Bob.
        rho00=copy(rho)
        rho01=copy(rho)
        rho10=copy(rho)
        rho11=copy(rho)

        # setting (A_0,B_0) -- phase shifts: (0,-π/4)
        rho00 = rho00 |> PS(0.0)(1) |> PS(-pi/4)(2)

        # setting (A_0,B_1) -- phase shifts: (0,π/4)
        rho01 = rho01 |> PS(0.0)(1) |> PS(pi/4)(2)

        # setting (A_1,B_0) -- phase shifts: (π/2,-π/4)
        rho10 = rho10 |> PS(pi/2)(1) |> PS(-pi/4)(2) 

        # setting (A_1,B_1) -- phase shifts: (π/2,π/4)
        rho11 = rho11 |> PS(pi/2)(1) |> PS(pi/4)(2) 
        
        
        # the sign binning:
        a=BitArray([1,0])
        b=a
        alpha=[-x,0,x]
        beta=[-y,0,y]
        
        return E(a,b,alpha,beta,rho00) + E(a,b,alpha,beta,rho01) + E(a,b,alpha,beta,rho10) - E(a,b,alpha,beta,rho11)
    end
end

#The global function whose weighted integral (where weights are the signs assigned by Alice and Bob) over V returns the CHSH value
function GlobalIntegrand(x,y,rho::PseudoGaussianState)
    rho00=copy(rho)
    rho01=copy(rho)
    rho10=copy(rho)
    rho11=copy(rho)
    # setting (A_0,B_0) -- phase shifts: (0,-π/4)
    rho00 = rho00 |> PS(0.0)(1) |> PS(-pi/4)(2)

    # setting (A_0,B_1) -- phase shifts: (0,π/4)
    rho01 = rho01 |> PS(0.0)(1) |> PS(pi/4)(2)

    # setting (A_1,B_0) -- phase shifts: (π/2,-π/4)
    rho10 = rho10 |> PS(pi/2)(1) |> PS(-pi/4)(2) 

    # setting (A_1,B_1) -- phase shifts: (π/2,π/4)
    rho11 = rho11 |> PS(pi/2)(1) |> PS(pi/4)(2) 

    V=[x;y]

    return abs(Integrand(V,rho00) + Integrand(V,rho01) + Integrand(V,rho10) - Integrand(V,rho11)) 
end

xs = range(-150, stop=150, length=30)
ys = range(-150, stop=150, length=30)

#c=contour(xs, ys, (x,y)->GlobalIntegrand(x,y,state), levels=15, color=:turbo, clabels=true)

#We define a function that checks if a CM is a valid CM of a Gaussian state. If one of the conditions is not met, the function tries to enforce it by correcting the CM. The function then outputs the Frobenius distance between the original matrix and the corrected matrix, as well as the corrected matrix. This function is based on the results of https://arxiv.org/pdf/0902.1502.pdf

function Gaussify!(rho::PseudoGaussianState)
    l=lastindex(rho.prob)
    for i in 1:l
        Gaussify!(rho.states[i])
    end
end

function Gaussify!(rho::GaussianState)
    gauss=true
    Mcopy=deepcopy(rho.σ)

    #here we enforce the CM to be symmetric
    M1=(Mcopy+transpose(Mcopy))/2
    
    
    
    #A symmetric CM is valid iff: 2σ + i Ω/2 >=0. We check this additional condition to strengthen the gaussification of the CM.
    commrel=true
    N=Int(size(Mcopy,1)/2)
    Ω=zeros(2*N,2*N)
    for m in 1:N
        Ω[2*m-1,2*m]=1.0
        Ω[2*m,2*m-1]=-1.0
    end
    evals2=eigvals(2*M1 + im*Ω/2)
    if evals2[1]<0  #we enforce nonnegative eigenvalues by adding max(|emin|,10^-16)*I/2 to the CM
        cm=deepcopy(M1)
        M1=cm + max(abs(evals2[1]),1e-16)*I/2
        commrel=false
    end
    pos=true
    sympl=true

    #=
    #A symmetric CM is a valid CM of a quantum state iff CM>0 && dk >= 1/2, where dk are its simplectic eigenvalues (note that pure Gaussian states satisfy: dk=1/2).
    pos=true
    evals=eigvals(M1)
    if evals[1]<0  # we enforce positive eigenvalues by adding max(2*|emin|,10^-16)*I to the CM, since the numerical error is 10^-16.
        cm=deepcopy(M1)
        M1=cm + max(2*abs(evals[1]),1e-16)*I
        pos=false
    end
    sympl=true
    D,S = williamson(2*M1)
    for k in 1:lastindex(D)
        if D[k]<0.5
            D[k]=0.5  # we enforce the symplectic eigenvalues to be >= 0.5
            sympl=false
        end
    end
    if !sympl
        Wdiag=wdiag(D)
        M1=transpose(S)*Wdiag*S/2
    end
    commrel=true
    =#

    rho.σ=(M1+transpose(M1))/2
    Frobenius=sqrt(tr(transpose(rho.σ-Mcopy)*(rho.σ-Mcopy)))
    gauss=pos && sympl && commrel

    return gauss, Frobenius

end

#The trace on a mode of a PseudoGaussian state
function traceout!(rho::PseudoGaussianState,mode::Int)
    mixnum=lastindex(rho.prob)
    for i in 1:mixnum
        ptrace!(rho.states[i],mode)
    end
end



####################################
####################################
####################################
#### SOME FUNCTIONS USED FOR TESTING

using QuantumOpticalCircuits
using HCubature


#This is the probability of obtaining two outcomes as the result of two X-quadrature measurements on Alice's and Bob's systems. It equals the integral of the 2-mode Wigner function over the P variables.
function Integrand(V,rho::PseudoGaussianState)
    w=rho.prob
    gaussrho=rho.states
    f=0
    for k in 1:length(w)
        if !issymmetric(gaussrho[k].σ)
            println("ERROR: the CM of the ", k,"-th Gaussian is not symmetric")
        end
        if !isposdef(gaussrho[k].σ)
            println("ERROR: the CM of the ", k,"-th Gaussian is not >0")
        end
        sigma=gaussrho[k].σ[[1,3],[1,3]]
        mu=gaussrho[k].d[[1,3],1]
        if !isposdef(sigma)
            println("ERROR: the CM of the ", k,"-th reduced Gaussian is not >0")
            return 2.0
        end
        invsigma=inv(sigma)
        sqd=sqrt(det(sigma))
        f += w[k]*exp(-(0.5)*(transpose(V-mu)*invsigma*(V-mu))[1])/(2*pi*sqd)
    end
    return f*rho.norm
end

#Because we observe that Integrand is negative for some values while it should not, we define a different version where we first combine the Wigner functions with the respective weights into PseudoWigner, and then we integrate numerically this over the P variables.
function PseudoWigner(X,rho::PseudoGaussianState)
    w=rho.prob
    gaussrho=rho.states
    wignermixture=0
    for k in 1:length(w)
        sigma=gaussrho[k].σ
        mu=gaussrho[k].d
        sqd=sqrt(det(sigma))
        invsigma=inv(sigma)

        wignermixture += w[k]*exp(-(0.5)*(transpose(X-mu)*invsigma*(X-mu))[1])/(4*(pi^2)*sqd)
    end
    return wignermixture*rho.norm
end

#It turns out that even IntegrandInverted can become negative on certain problematic states
function IntegrandInverted(V,rho::PseudoGaussianState)
    f(V,rho,Y)=PseudoWigner([V[1],Y[1],V[2],Y[2]],rho)
    return hcubature(Y -> f(V,rho,Y), (-150,-150), (150,150); rtol=1e-7, atol=0)[1]
end


#Here we compute the expectation value by exchanging the order of integration and weighted sum
function Einverted(a::BitVector,b::BitVector,alpha::Vector,beta::Vector,rho::PseudoGaussianState)
          
    summand=0
    
    for r in 1:lastindex(a)
        for s in 1:lastindex(b)
            if a[r] - b[s] == 0
                x0=alpha[r]
                x1=alpha[r+1]
                y0=beta[s]
                y1=beta[s+1]
                summand += hcubature(V -> Integrand(V,rho), (x0,y0), (x1,y1); rtol=1e-6, atol=0)[1]
            end
        end
    end
    
    return 2*summand-1
end


#Here we plot the function Integrand, which is supposed to be a probability distribution. 
using QuantumOpticalCircuits
using Plots; pythonplot()


function f(x,y)
    initstate=PseudoGaussianState(4) |> BS(1.14409)(2,4) |> TMS(0.92463)(1,2) |> PS(0.42305)(2) |> BS(1.5664)(1,4) |> TMS(0.30906)(3,4) |> PS(0.52058)(2) |> BS(1.5341)(1,4) |> PS(0.39973)(1) |> TMS(1.039)(1,2) |> PS(0.3215)(3) |> BS(2.20448)(1,2) |> PS(0.89547)(2) |> BS(0.01483)(1,4) |> BS(3.12806)(2,3) |> TMS(0.01138)(1,3) |> Heralding(4,η=1.0,tol=1e-10,onclick=true) |> Heralding(3,η=1.0,tol=1e-10,onclick=true) |> PS(0.0)(1) |> PS(-pi/4)(2)
    Gaussify!(initstate)
    Gaussify!(initstate)

    state=initstate

    v=[x ; y]

    return Integrand(v,state)
end

function fCorentine(x,y)
    state=PseudoGaussianState(4) |> BS(1.14409)(2,4) |> TMS(0.92463)(1,2) |> PS(0.42305)(2) |> BS(1.5664)(1,4) |> TMS(0.30906)(3,4) |> PS(0.52058)(2) |> BS(1.5341)(1,4) |> PS(0.39973)(1) |> TMS(1.039)(1,2) |> PS(0.3215)(3) |> BS(2.20448)(1,2) |> PS(0.89547)(2) |> BS(0.01483)(1,4) |> BS(3.12806)(2,3) |> TMS(0.01138)(1,3) |> Heralding(4,η=1.0,tol=1e-10,onclick=true) |> Heralding(3,η=1.0,tol=1e-10,onclick=true)

    v=[x ; y]

    return IntegrandCorentine(v,state)
end

xs = range(-300, stop=300, length=50)
ys = range(-300, stop=300, length=50)

#c=contour(xs, ys, f, levels=15, color=:turbo, clabels=true)

######################################
### IMPORTED FROM CORENTINE'S CODE ###
######################################
######################################

mutable struct GaussianState_full <: AbstractState

    d::Vector{Float64}
	σ::Matrix{Float64}
    icov::Matrix{Float64} #the inverse of sigma
    C::Float64 # the constant in front of :Union(Wigner,PseudoWigner)

end

mutable struct PseudoGaussianState_full <: AbstractState
    norm::Float64
    prob::Vector{Float64}
	states::Vector{GaussianState_full}

end

function Cvrt(Gstate::GaussianState) #Convert Gaussianstates into Gaussianstate_full

    #return GaussianState_full(Gstate.d, Gstate.σ, inv(Gstate.σ) , 1/(sqrt(( (2*pi)^k )*det(Gstate.σ))) )
    return GaussianState_full(Gstate.d, Gstate.σ, inv(Gstate.σ) , 1/(sqrt(det(2*pi*Gstate.σ))) )

end

function Cvrt(PGstate::PseudoGaussianState)

    M = size(PGstate.prob)[1] #Number of gaussian states into the pseudo gaussian state
    states = Vector{GaussianState_full}(undef,M)

    for i in range(1,M)
        states[i]=Cvrt(PGstate.states[i])
    end

    
    return PseudoGaussianState_full(PGstate.norm,PGstate.prob,states)
end


function integrate_p(Gstate::GaussianState_full) #only for 2 modes for now

    S_hom = Matrix{Float64}(undef,4,4) #make (x1,p1,x2,p2) -> (x1,x2,p1,p2) it is the right formula
    fill!(S_hom,0)
    S_hom[1,1]=1
    S_hom[2,3]=1
    S_hom[3,2]=1
    S_hom[4,4]=1


    icov = S_hom * Gstate.icov * transpose(S_hom) #transpose is useless 
    d = Vector{Float64}(undef,2)

    d[1]=Gstate.d[1] #displacement not modified (x1,x2)
    d[2]=Gstate.d[3]

    A = Matrix{Float64}(undef,2,2) #define the matrix
    C = Matrix{Float64}(undef,2,2)
    B = Matrix{Float64}(undef,2,2)

    A=icov[1:2,1:2] #extract submatrices
    C=icov[1:2,3:4]
    B=icov[3:4,3:4]

    icov_reduced = A-C*inv(B)*transpose(C)
    σ_reduced =  inv(icov_reduced)

    
    return GaussianState_full(d, σ_reduced , icov_reduced, 1/(sqrt(det(2*pi*σ_reduced)))  )

end

function integrate_p(PGstate::PseudoGaussianState_full) #only for 2 modes for now

    M = size(PGstate.prob)[1] #Number of gaussian states into the pseudo gaussian state
    states = Vector{GaussianState_full}(undef,M)

    for i in range(1,M)
        states[i]=integrate_p(PGstate.states[i])
    end

    
    return PseudoGaussianState_full(PGstate.norm,PGstate.prob,states)

end

#This is the probability of obtaining two outcomes as the result of two X-quadrature measurements on Alice's and Bob's systems. It equals the integral of the 2-mode Wigner function over the P variables. We verified that it equals the function Integrand defined above via my reduction to the X variables.
function IntegrandCorentine(V,rho::PseudoGaussianState)

    rhocv=Cvrt(rho)
    intrho=integrate_p(rhocv)

    M = length(intrho.prob) #number of Gaussian states into the pseudo gaussian state

    sum = 0

    for i in 1:M
        mu=intrho.states[i].d
        sigma=intrho.states[i].σ
        sqd=sqrt(det(sigma))
        invsigma=intrho.states[i].icov
        sum+= intrho.prob[i]*exp(-(0.5)*(transpose(V-mu)*invsigma*(V-mu))[1])/(2*pi*sqd)
    end

    return sum*intrho.norm
    

end
######################################
########### END OF IMPORT ############
######################################
######################################




#Here we explicitly compute the negative contributions to the expectation value instead of using the fact that the state is normalized. We also check that the integral of a normalized Guassian is always between 0 and 1.
function Enew(a::BitVector,b::BitVector,alpha::Vector,beta::Vector,rho::PseudoGaussianState)
    summand=0
    w=rho.prob
    gaussrho=rho.states
    
    for k in 1:length(w)
        sigma=gaussrho[k].σ[[1,3],[1,3]]  
        mu=gaussrho[k].d[[1,3],1]
        sc=sigma[1,2]
        sb=sigma[2,2]
        sqd=sqrt(det(sigma))
        
        summand2=0
        summand3=0

        for r in 1:lastindex(a)
            for s in 1:lastindex(b)
                x0=sqrt(sb)*(alpha[r]-mu[1])/sqrt(2)
                x1=sqrt(sb)*(alpha[r+1]-mu[1])/sqrt(2)
                y0=(beta[s]-mu[2])/sqrt(2*sb)
                y1=(beta[s+1]-mu[2])/sqrt(2*sb)
                integ=PartialInt(sc,sqd,x0,x1,y0,y1)/2
                if integ>1 || integ<0
                    println("ERROR: integral of normalized Gaussian is outside [0,1]: ",integ)
                end
                if a[r] - b[s] == 0
                    summand2 += integ
                else
                    summand3 += integ
                end
            end
        end
        summand +=  w[k]*(summand2-summand3)
        println(summand)
    end
    summand = summand*rho.norm
    return summand
end

function CHSHnew(rho::PseudoGaussianState)
    
    if rho.norm > 1e12 #if the norm is too big, the numerical imprecision of the numerical integrals (tolerance is 1e-12) spoils the calculation of E
        return 0
    else
        #we create 4 copies of the circuit state, one for each setting combination. Then we process each copy of the state according to the settings chosen by Alice and Bob.
        rho00=copy(rho)
        rho01=copy(rho)
        rho10=copy(rho)
        rho11=copy(rho)

        # setting (A_0,B_0) -- phase shifts: (0,-π/4)
        rho00 = rho00 |> PS(0.0)(1) |> PS(-pi/4)(2)

        # setting (A_0,B_1) -- phase shifts: (0,π/4)
        rho01 = rho01 |> PS(0.0)(1) |> PS(pi/4)(2)

        # setting (A_1,B_0) -- phase shifts: (π/2,-π/4)
        rho10 = rho10 |> PS(pi/2)(1) |> PS(-pi/4)(2) 

        # setting (A_1,B_1) -- phase shifts: (π/2,π/4)
        rho11 = rho11 |> PS(pi/2)(1) |> PS(pi/4)(2) 
        
        
        # the sign binning:
        a=BitArray([1,0])
        b=a
        alpha=[-50,0,50]
        beta=alpha
        
        
        #=
        # a new binning inspired by the results of Xavier
        a=BitArray([1,0,1])
        b=a
        alpha=[-50,-0.8886,0.8886,50]
        beta=alpha
        =#

        return Enew(a,b,alpha,beta,rho00) + Enew(a,b,alpha,beta,rho01) + Enew(a,b,alpha,beta,rho10) - Enew(a,b,alpha,beta,rho11)
    end
end



# Here we replace the probability function with a constant function equal to 1. We expect that the output of E is: 2*((alpha[3]-alpha[2])*(beta[3]-beta[2])+(alpha[4]-alpha[3])*(beta[4]-beta[3]) + (alpha[2]-alpha[1])*(beta[4]-beta[3]) + (alpha[2]-alpha[1])*(beta[2]-beta[1]) + (alpha[4]-alpha[3])*(beta[2]-beta[1]))-1

function Intconst(x0,x1,y0,y1)
    f(V)=1
    return hcubature(f, (x0,y0), (x1,y1); rtol=1e-12, atol=0)[1]
end




function CHSHopt(rho::PseudoGaussianState)
    
    newnorm=1/sum(rho.prob)
    if (abs(newnorm-rho.norm) > newnorm*1e-3) || (newnorm > 1e12) # if the relative difference between the normalization of the state and the alternative normalization (given by the sum of the weights of the Gaussian mixture) is too large, we reject the state because it means that the Gaussian states in the mixture are not properly normalized. Likewise, if the norm is too big, the numerical imprecision of the numerical integrals (tolerance is 1e-12) spoils the calculation of E
        return 0
    else
        #we create 4 copies of the circuit state, one for each setting combination. Then we process each copy of the state according to the settings chosen by Alice and Bob.
        rho00=copy(rho)
        rho01=copy(rho)
        rho10=copy(rho)
        rho11=copy(rho)

        # setting (A_0,B_0) -- phase shifts: (0,-π/4)
        rho00 = rho00 |> PS(0.0)(1) |> PS(-pi/4)(2)

        # setting (A_0,B_1) -- phase shifts: (0,π/4)
        rho01 = rho01 |> PS(0.0)(1) |> PS(pi/4)(2)

        # setting (A_1,B_0) -- phase shifts: (π/2,-π/4)
        rho10 = rho10 |> PS(pi/2)(1) |> PS(-pi/4)(2) 

        # setting (A_1,B_1) -- phase shifts: (π/2,π/4)
        rho11 = rho11 |> PS(pi/2)(1) |> PS(pi/4)(2) 
        
               
        # here we optimize the binning width
        a=BitArray([1,0,1])
        b=a
        alpha(x)=[-50,-x,x,50]
        
        function f(x)
            vec=alpha(x[1])
            return -abs(E(a,b,vec,vec,rho00) + E(a,b,vec,vec,rho01) + E(a,b,vec,vec,rho10) - E(a,b,vec,vec,rho11))
        end
        
        
        ep=1e-2
        lower=[-50+ep,0]
        upper=[50-ep,10]
        initial_x=[0.88,5]

        m = optimize(f,lower,upper,initial_x, NelderMead(), Optim.Options(time_limit=100))
        xopt=Optim.minimizer(m)

        return -f(xopt) , xopt[1]
    end
end



#Here we define a function that computes the integral of a Gaussian in 2 dimensions, taking as inputs the intervals for the two integration variables and the matrix A in exp[-(1/2)V^T A V].
function GaussIntold(x0,x1,y0,y1,A)
    f(V)=exp(-(0.5)*(transpose(V)*A*(V))[1])
    hcubature(f, (x0,y0), (x1,y1); rtol=1e-12, atol=0)[1]
end

#Here we define a function that computes the integral of a Gaussian in 2 dimensions, taking as inputs the intervals for the two integration variables and the matrix A in exp[-(1/2)V^T A V]. We compared the speed of this function with the the approach of 2d numerical integration with hcubature (measures_homodyne_v3, GaussIntold) and this is much faster!
function GaussInt(x0,x1,y0,y1,A)
    r=A[1,1]
    s=A[2,2]
    t=A[1,2]
    f(x)=sqrt(pi/r)*exp(-(1/r)*(r*s-t^2)*x^2)*(erf(sqrt(r/2)*x1 + (t/sqrt(r))*x) - erf(sqrt(r/2)*x0 + (t/sqrt(r))*x))
    return quadgk(f, y0/sqrt(2), y1/sqrt(2), rtol=1e-12, atol=0)[1]
end

# We insert a check that the integral of a normalized Gaussian is always between 0 and 1, otherwise an error is returned; we also check that each sigma is >0.
function Eold(a::BitVector,b::BitVector,alpha::Vector,beta::Vector,rho::PseudoGaussianState)
    summand=0
    w=rho.prob
    gaussrho=rho.states
    gaussnorm=1/sum(w)
    
    for k in 1:length(w)
        sigma=gaussrho[k].σ[[1,3],[1,3]]
        mu=gaussrho[k].d[[1,3],1]
        if !isposdef(sigma)
            return println("matrix sigma is not positive definite")
        end

        evals=real(eigvals(sigma))  #to avoid complex numbers due to numerical errors
        emax=maximum(evals)
        emin=minimum(evals)
        invsigma=inv(sigma)*emax
            
        summand2=0

        for r in 1:length(a)
            for s in 1:length(b)
                if a[r] - b[s] == 0
                    x0=(alpha[r]-mu[1])/sqrt(emax)
                    x1=(alpha[r+1]-mu[1])/sqrt(emax)
                    y0=(beta[s]-mu[2])/sqrt(emax)
                    y1=(beta[s+1]-mu[2])/sqrt(emax)
                    gaussint=GaussInt(x0,x1,y0,y1,invsigma)/(2*pi)
                    if gaussint<=0 || gaussint>1
                        return println("ERROR: the integral of a normalized Gaussian is outside the interval [0,1]")
                    else
                        summand2 += gaussint
                    end
                end
            end
        end
        summand +=  w[k]*sqrt(emax/emin)*summand2
    end
    summand = 2*summand*gaussnorm-1
    return summand
end

function CHSHold(rho::PseudoGaussianState)
    
    newnorm=1/sum(rho.prob)
    if (abs(newnorm-rho.norm) > newnorm*1e-3) || (newnorm > 1e12) # if the relative difference between the normalization of the state and the alternative normalization (given by the sum of the weights of the Gaussian mixture) is too large, we reject the state because it means that the Gaussian states in the mixture are not properly normalized. Likewise, if the norm is too big, the numerical imprecision of the numerical integrals (tolerance is 1e-12) spoils the calculation of E
        return 0
    else
        #we create 4 copies of the circuit state, one for each setting combination. Then we process each copy of the state according to the settings chosen by Alice and Bob.
        rho00=deepcopy(rho)
        rho01=deepcopy(rho)
        rho10=deepcopy(rho)
        rho11=deepcopy(rho)

        # setting (A_0,B_0) -- phase shifts: (0,-π/4)
        rho00 = rho00 |> PS(0.0)(1) |> PS(-pi/4)(2)

        # setting (A_0,B_1) -- phase shifts: (0,π/4)
        rho01 = rho01 |> PS(0.0)(1) |> PS(pi/4)(2)

        # setting (A_1,B_0) -- phase shifts: (π/2,-π/4)
        rho10 = rho10 |> PS(pi/2)(1) |> PS(-pi/4)(2) 

        # setting (A_1,B_1) -- phase shifts: (π/2,π/4)
        rho11 = rho11 |> PS(pi/2)(1) |> PS(pi/4)(2) 
        
        #=
        # the sign binning:
        a=BitArray([1,0])
        b=a
        alpha=[-50,0,50]
        beta=alpha
        =#
        
        # a new binning inspired by the results of Xavier
        a=BitArray([1,0,1])
        b=a
        alpha=[-50,-0.8886,0.8886,50]
        beta=alpha
        

        return Eold(a,b,alpha,beta,rho00) + Eold(a,b,alpha,beta,rho01) + Eold(a,b,alpha,beta,rho10) - Eold(a,b,alpha,beta,rho11)
    end
end


#Here we just check that the PseudoGaussian state is normalized by computing its trace

function TwomodeGaussInt(A)
    f(V)=exp(-(0.5)*(transpose(V)*A*(V))[1])
    sqrt(det(A))*hcubature(f, (-50,-50,-50,-50), (50,50,50,50); rtol=1e-9, atol=0)[1]/((2*pi)^2)
end

function trace(rho::PseudoGaussianState)
    summand=0
    w=rho.prob
    gaussrho=rho.states
    
    for k in 1:length(w)
        if !isposdef(gaussrho[k].σ)
            println("the CV matrix of the ", k, "-th state is not positive definite:")
            return println(gaussrho[k].σ)
        end
        invsigma=inv(gaussrho[k].σ)
        gaussint=TwomodeGaussInt(invsigma)
        if !(1-1e-9<= gaussint <= 1+1e-9)
            println("ERROR: the integral of a normalized Gaussian is outside the interval [1-10^-9,1+10^-9]")
            return println("number of state in the mixture: ",k,"; gaussint=",gaussint,"; ")
        else
            summand +=  w[k]*gaussint
        end
    end
    summand = summand*rho.norm
    return summand
end

