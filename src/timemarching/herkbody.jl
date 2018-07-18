include(Pkg.dir("Dyn3d")*"/src/Dyn3d.jl")
using Dyn3d

"""
    HERKBody()

HERKBody is a half-explicit Runge-Kutta solver based on the
constrained body model in paper of V.Brasey and E.Hairer.
The following ODE system is being solved:
   | dq/dt = v                     |
   | M(q)*dv/dt = f(q,v) - GT(q)*λ |
   | 0 = G(q)*v + gti(q)           |
, where GT is similar to the transpose of G.

Note that this is a index-2 system with two variables q and u.
Here we denote for a body chain, v stands for body velocity
and vJ stands for joint velocity. Similarly for q and qJ. λ is the
constraint on q to be satisfied.

Specificlly for the dynamics problem, we choose to solve for qJ and v.
The system of equation is actually:
   | dqJ/dt = vJ                              |
   | M(qJ)*dv/dt = f(qJ,v,vJ) - GT(qJ)*lambda |
   | 0 = G(qJ)*v + gti(qJ)                    |
So we need a step to calculate v from vJ solved. The motion constraint
(prescribed active motion) is according to joint, not body.
"""
mutable struct HERKBody{NS,FA,FB1,FB2,FR1,FR2,TU,TF}

  # rk parameters
  rk :: RKParams

  # body dynamics state
  body_dyn :: BodyDyn

  # left hand side matrix components
  A :: FA # operates on TU and returns TF
  B₁ᵀ :: FB1  # operates on TF and returns TU
  B₂ :: FB2   # operates on TU and returns TF
  r₁ :: FR1  # function of u and t, returns TU
  r₂ :: FR2  # function of t, returns TF

  # functions to update lhs matrix components
  UpP :: FP # update joint.qJ
  UpV :: FV # update body.v

  # Saddle-point systems
  S :: Vector{SaddleSystem}  # -B₂AB₁ᵀ

  # scratch space
  ubuffer :: TU
  fbuffer :: TF

  # flags
  _issymmetric :: Bool

  # tolerance
  tol :: Float64

end

function (::Type{HERKBody})(body_dyn::BodyDyn,
                            A::FA,B₁ᵀ::FB1,B₂::FB2,r₁::FR1,r₂::FR2,
                            UpP::FP,UPV::FV;
                            issymmetric::Bool=false,
                            rk::RKParams{NS}=RK31,tol::Float64=1e-4
                            ) where {TU,TF,FA,FB1,FB2,FR1,FR2,NS}

    # scratch space
    ubuffer = deepcopy(u)
    w = [deepcopy(u) for i = 1:NS-1]
    fbuffer = deepcopy(f)

    dclist = diff([0;rk.c])

    # construct an array of operators for the integrating factor. Each
    # one can act on data of type `u` and return data of the same type.
    # e.g. we can call Hlist[1](u) to get the result.
    Hlist = [u -> plan_intfact(dc*Δt,u)*u for dc in unique(dclist)]

    Slist = [SaddleSystem(u,f,H,B₁ᵀ,B₂,issymmetric=issymmetric,isposdef=true) for H in Hlist]

    H = [Hlist[i] for i in indexin(dclist,unique(dclist))]
    S = [Slist[i] for i in indexin(dclist,unique(dclist))]

    htype,_ = typeof(H).parameters

    HERKBody{NS,FA,FB1,FB2,FR1,FR2,TU,TF}(rk,body_dyn
                                A,B₁ᵀ,B₂,r₁,r₂,UpP,Upv,S,
                                ubuffer,fbuffer,
                                issymmetric,tol)
end

function Base.show(io::IO, scheme::HERKBody{NS,FA,FB1,FB2,FR1,FR2,TU,TF}) where {NS,FA,FB1,FB2,FR1,FR2,TU,TF}
    println(io, "Order-$NS HERK-Body system with")
    println(io, "   State of type $TU")
    println(io, "   Force of type $TF")
end

function (scheme::HERKBody{NS,FA,FB1,FB2,FR1,FR2,TU,TF})(soln::Soln{TS}) where {NS,FA,FB1,FB2,FR1,FR2,TU,TF,TS}

    @get scheme (Δt,rk,body_dyn,A,S,B₁ᵀ,B₂,r₁,r₂,fbuffer,ubuffer,tol)
    @get body_dyn (bs, js, sys)

    qJ_dim = sys.ndof
    λ_dim =sys.ncdof_HERK

    # pointer to pre-allocated array
    @get sys.pre_array (qJ, vJ, v, v̇, λ, v_temp, Mᵢ₋₁, fᵢ₋₁, GTᵢ₋₁, Gᵢ, gtiᵢ,
        lhs, rhs)

    # stage 1
    tᵢ₋₁ = soln.t; tᵢ = soln.t;; dt = soln.dt
    qJ[1,:] = sᵢₙ.qJ
    v[1,:] = sᵢₙ.v
    # update vJ using v
    bs, js, sys, vJ[1,:] = UpV(bs, js, sys, v[1,:])

    # stage 2 to st+1
    for i = 2:NS+1
        # time of i-1 and i
        tᵢ₋₁ = tᵢ
        tᵢ = sᵢₙ.t + dt*rk.c[i-1]
        # initialize qJ[i,:]
        qJ[i,:] = sᵢₙ.qJ
        # calculate M, f and GT at tᵢ₋₁
        Mᵢ₋₁ = FA(body_dyn)
        fᵢ₋₁ = FR1(body_dyn)
        GTᵢ₋₁ = FB1(body_dyn)
        # advance qJ[i,:]
        for k = 1:i-1
            qJ[i,:] += dt*rk.a[i-1,k]*view(vJ,k,:)
        end
        # use new qJ to update system position
        bs, js, sys = UpP(bs, js, sys, qJ[i,:])
        # calculate G and gti at tᵢ
        Gᵢ = FB2(body_dyn)
        gtiᵢ = FR2(tᵢ, body_dyn)

#        Sᵢ = SaddleSystem((u,f),(H,B₁ᵀ,B₂),issymmetric=issymmetric,isposdef=true)

        # construct lhs matrix
        lhs = [ Mᵢ₋₁ GTᵢ₋₁; Gᵢ zeros(T,λ_dim,λ_dim) ]
        # the accumulated v term on the right hand side
        v_temp = sᵢₙ.v
        for k = 1:i-2
            v_temp += dt*rk.a[i-1,k]*view(v̇,k,:)
        end
        # construct rhs
        rhs = [ fᵢ₋₁; -1./(dt*rk.a[i-1,i-1])*(Gᵢ*v_temp + gtiᵢ) ]
######### use Julia's built in "\" operator for now
        # solve the eq
        x = lhs \ rhs
        # x = BlockLU(lhs, rhs, qJ_dim, λ_dim)
        # apply the solution
        v̇[i-1,:] = x[1:qJ_dim]
        λ[i-1,:] = x[qJ_dim+1:end]
        # advance v[i,:]
        v[i,:] = sᵢₙ.v
        for k = 1:i-1
            v[i,:] += dt*rk.a[i-1,k]*view(v̇,k,:)
        end
        # update vJ using updated v
        bs, js, sys, vJ[i,:] = UpdateVelocity!(bs, js, sys, v[i,:])
# println("v = ", v[i,:])
    end

    # use norm(v[st+1,:]-v[st,:]) to determine next timestep
    sₒᵤₜ = Soln(tᵢ) # init struct
    sₒᵤₜ.dt = sᵢₙ.dt*(tol/norm(view(v,st+1,:)-view(v,st,:)))^(1/3)
    sₒᵤₜ.t = sᵢₙ.t + sᵢₙ.dt
    sₒᵤₜ.qJ = view(qJ, st+1, :)
    sₒᵤₜ.v = view(v, st+1, :)
    sₒᵤₜ.v̇ = view(v̇, st, :)
    sₒᵤₜ.λ = view(λ, st, :)

    return  sₒᵤₜ, bs, js, sys
end
