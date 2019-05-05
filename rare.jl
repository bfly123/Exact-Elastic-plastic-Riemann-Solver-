
function Presolve(varL::Var, varR::Var, conL::Const,conR::Const)
        
    ρL, uL, pL, sxxL = varL.ρ, varL.u, varL.p, varL.sxx
    ρR, uR, pR, sxxR = varR.ρ, varR.u, varR.p, varR.sxx
    σL  = -pL+sxxL
    
    UL = [ρL, uL, pL, sxxL]
    UR = [ρR, uR, pR, sxxR]
    cL = sound(UL[:],conL)
   # eL = PToe(ρL, pL, conL)
    
    σR  = -pR+sxxR
    cR = sound(UR[:],conR)
  #  eR = PToe(ρR, pR, conR)
    
    sL = min(uL-cL, uR-cR)
    sR = max(uL+cL, uR+cR)
        
    s_star = (σL - σR + ρL*uL*(sL-uL) - ρR* uR*(sR-uR))/(ρL*(sL-uL)-ρR*(sR-uR))
        
    ρLstar = ρL*(uL-sL)/(s_star - sL)
    ρRstar = ρR*(uR-sR)/(s_star - sR)
        
    sxxLstar =  sxxL -4/3*conL.μ *(log(ρLstar) - log(ρL))
    sxxRstar =  sxxR -4/3*conR.μ *(log(ρRstar) - log(ρR))
 #   @show sxxLstar, sxxRstar,ρLstar,ρRstar
    caseL = CaseSelect(sxxL,sxxLstar, ρL, ρLstar,conL,1)
    caseR = CaseSelect(sxxR,sxxRstar, ρR, ρRstar,conR,2)
    return caseL, caseR
end

function CaseSelect(sxx₀::Float64, sxx₁::Float64,ρ₀::Float64, ρ₁::Float64,con::Const,LoR::Int)
    Y0 = con.Y0
      if abs(sxx₁) ≥ 2/3*Y0  && abs(sxx₀) < 2/3*Y0   
        if ρ₁ > ρ₀
            if LoR == 1
                case = Shock(2) # 左侧双激波
            else
                case = Shock(4) # 右侧双激波 
            end
        else
            if LoR == 1
                case = Rare(2) # case2 Left
            else
                case = Rare(4) # case2 Right
            end
        end
    else
        if ρ₁ > ρ₀
            if LoR == 1
                case = Shock(1) # case3
                else
                case = Shock(3)
            end
        else
            if LoR == 1
                case = Rare(1) #case1 Left
            else
                case = Rare(3)
            end
        end
    end
    return case
end

struct Shock
    cs::Int
end

struct Rare
    cs::Int
    
end
##case 说明  cs=1 为左单波 cs=2 为左双波
##          cs=3 为右单波 cs= 4 为右双波

function Sxx(ρ, var0::Var, con::Const)
    sxx0,ρ0 = var0.sxx,var0.ρ
    Y0   = con.Y0
    
    sxx = sxx0 -4/3*con.μ *(log(ρ) - log(ρ0))
    if abs(sxx) ≥ 2/3*Y0
        sxx = 2/3*Y0*sign(sxx)
    end
    return sxx
end

function fp(ρ::Float64, var0::Var, con::Const, case::Rare)
    ρ₀,p₀ = var0.ρ,var0.p
    ρ0,Γ0,a0= con.ρ0,con.Γ0,con.a0   
    λ₁ = ρ0*Γ0
    f2(ρ) =a0^2*fηη(ρ, con) -λ₁*Sxx(ρ, var0,con)/ρ^2

    p = p₀*exp(λ₁/ρ₀ - λ₁/ρ) + exp(-λ₁/ρ)*GaussIntegral(ρ->
       f2(ρ)*exp(λ₁/ρ), ρ₀, ρ,5)
    return p
end
#pLStar = fp(ρLStar,varL,con)

function fu(ρ::Float64, var0::Var, con::Const,case::Rare)
    ρ₀,u₀ = var0.ρ, var0.u
    a0,ρ0,Γ0,μ= con.a0, con.ρ0, con.Γ0, con.μ
    λ₁ = ρ0*Γ0
    S2(ρ) = a0^2*fηη(ρ,con)+λ₁*(fp(ρ,var0,con,case)-Sxx(ρ,var0,con))/ρ^2
    function f2(ρ)
        if case.cs == 1 &&  case.cs ==2
            return -√(S2(ρ)+4con.μ/(3ρ)) /ρ
        else
            return √(S2(ρ)+4con.μ/(3ρ)) /ρ
        end
    end
    
    u = u₀ + GaussIntegral(ρ->f2(ρ), ρ₀, ρ, 5)
    return u
end

function GaussIntegral(f::Function,x₀::Float64,x₁::Float64,order::Int)
    t₁= (x₁-x₀)/2
    t₂= (x₁+x₀)/2
    ω = zeros(Float64, 5)
    p = zeros(Float64, 5)
    
    if order == 1
        ω[1] = 2.0
        p[1] = 0.0
    elseif order == 3
        ω[1] = 1.0; ω[2] = 1.0
        p[1] = 1/√3.0; p[2] = -1/√3.0
    elseif order == 5
        ω[1] = 8.0/9; ω[2] = 5.0/9; ω[3] = 5.0/9
        p[1] = 0.0; p[2] = -√(3.0/5); p[3] = √(3.0/5)
    elseif order == 7
        ω[1] = (18+√30)/36; ω[2] = (18+√30)/36
        ω[3] = (18-√30)/36; ω[4] = (18-√30)/36
        p[1] = √(3/7-2/7*√(6/5)); p[1] = -√(3/7-2/7*√(6/5))
        p[3] = √(3/7+2/7*√(6/5)); p[4] = -√(3/7+2/7*√(6/5))
    end
    ∑ =sum( t₁*ω[i]*f(t₁*p[i]+t₂) for i in 1:floor(Int,order/2)+1)

    return ∑
end

function Ṽar(var0::Var, case, con::Const)

    Y0, μ= con.Y0, con.μ
    sxx,ρ = var0.sxx,var0.ρ

    cs = case.cs
    if cs == 1 || cs == 3
        var1 = var0
    elseif cs == 2 || cs == 4
        
        if typeof(case) == Rare 
            sxx1 = 2/3*Y0
            ρ1 = ρ*exp(-Y0/(2μ)+(3sxx)/(4μ))
        elseif typeof(case) == Shock
            sxx1 = -2/3*Y0
            ρ1 = ρ*exp(Y0/(2μ)+(3sxx)/(4μ))
        end
        p1 = fp(ρ, var0, con, case)
        u1 = fu(ρ, var0, con, case)
    
        var1= Var(ρ1, u1, p1, sxx1)
    end
    return var1
end


function fp(ρ::Float64, var0::Var, con::Const,case::Shock)
   
    ρ₀,p₀,sxx₀ = var0.ρ,var0.p,var0.sxx
    ρ0,Γ0,a0 = con.ρ0,con.Γ0,con.a0
    
    σ₀ = -p₀+sxx₀
    e₀ =  PToe(ρ₀, p₀, con)
    
    t =  ρ₀*ρ/(ρ-ρ₀) ; c₀ = 1/(Γ0*ρ0); c₁= a0^2/Γ0
    p = (2t*(c₁*fη(ρ) + e₀) -(σ₀+Sxx(ρ,var0,con))
         )/(1-2t*c₀)
    
    return p
end

function fu(ρ::Float64, var0::Var, con::Const,case::Shock)
   
    ρ₀,u₀,p₀,sxx₀ = var0.ρ,var0.u,var0.p,var0.sxx
    σ₀ = -p₀+sxx₀
    
    sxx=Sxx(ρ,var0,con)
    p  = fp(ρ, var0, con, case)
    σ = -p+sxx
    
    t =  ρ₀*ρ/(ρ-ρ₀)
    if case.cs == 1 || case.cs == 2
        u = u₀ - √((σ₀ - σ)/t)
    else
        u = u₀ + √((σ₀ - σ)/t)
    end
    return u
end

    fL(ρ) = fu(ρ, var0L, conL, caseL)
    fR(ρ) = fu(ρ, var0R, conR, caseR)
    gL(ρ) = -fp(ρ,var0L,conL,caseL)+Sxx( ρ, var0L, conL)
    gR(ρ) = -fp(ρ,var0R,conR,caseR)+Sxx( ρ, var0R, conR)

    fL¹(ρ) = Derivative(fL,ρ)
    fR¹(ρ) = Derivative(fR,ρ)
    gL¹(ρ) = Derivative(gL,ρ)
    gR¹(ρ) = Derivative(gR,ρ)

function Derivative(f::Function,x::Float64)
    ϵ₀ = 1e-8
    if abs(x) >= ϵ₀
        ϵ= ϵ₀*x
    else
        ϵ =ϵ₀
    end
    f¹(x) = (f(x+ϵ)-f(x))/ϵ
    return f¹(x)
end

function NewtonIter!(U::Array{Float64,1},J::Array{Float64,2},F::Array{Float64,1}, TOL::Float64 = 1.e-6) 
    I, = size(U)
    U_new = zeros(Float64, I)
    while true  
        c = max(abs(F[1]),abs(F[2]))
        if c <= TOL
            break
        end
        U_new = U - inv(J)*F # same to inv(J)×F
   #     c= CHA(U,U_new,F)
     #   if c <= TOL
      #      break
      #  end
    end
    return U
    
end   

function CHA(R0::Array{Float64,1}, R1::Array{Float64,1}, F::Array{Floa64,1})
    
    ρₗ₀ = R0[1]; ρᵣ₀ = R0[2]
    ρₗ₁ = R1[1]; ρᵣ₁ = R1[2]
    
    CHA = max(abs(2(ρₗ₁-ρₗ₀)/(ρₗ₁+ρₗ₀)),
              abs(2(ρᵣ₁-ρᵣ₀)/(ρᵣ₁+ρᵣ₀)), F[1], F[2])
    return CHA
end   

include("/home/bfly/workspace/Juliastudy/Lagrangian_1d-2materials.jl")

function Exact_Riemann(uL::Array{Float64,1}, uR::Array{Float64,1}, conL::Const, conR::Const )
        
    
    var0L = Var(uL[1],uL[2],uL[3],uL[4])
    var0R = Var(uR[1],uR[2],uR[3],uR[4])
    @show var0L
    
###预求解和分类    
    caseL,caseR = Presolve(var0L, var0R, conL, conR)
    @show caseL, caseR
###求解  Q̂ 
    varL = Ṽar(var0L, caseL,conL)
    varR = Ṽar(var0R, caseR,conR)
    ρL, uL, pL, sxxL = varL.ρ, varL.u, varL.p, varL.sxx
    ρR, uR, pR, sxxR = varR.ρ, varR.u, varR.p, varR.sxx

    ###迭代初始化    
    ρLStar = (ρL +ρR)/2
    ρRStar = (ρL +ρR)/2
   
    fL(ρ) = fu(ρ, varL, conL, caseL)
    fR(ρ) = fu(ρ, varR, conR, caseR)
    gL(ρ) = -fp(ρ,varL,conL,caseL)+Sxx( ρ, varL, conL)
    gR(ρ) = -fp(ρ,varR,conR,caseR)+Sxx( ρ, varR, conR)

    fL¹(ρ) = Derivative(fL,ρ)
    fR¹(ρ) = Derivative(fR,ρ)
    gL¹(ρ) = Derivative(gL,ρ)
    gR¹(ρ) = Derivative(gR,ρ)
    
    TOL = 1.e-6
    while true
      
        J = [[-fR¹(ρRStar) fL¹(ρLStar)]; 
             [-gR¹(ρRStar) gL¹(ρLStar)]]
    
        U = [ρRStar, ρLStar]
    
        F = [fL(ρLStar)- fR(ρRStar), 
             gL(ρLStar)- gR(ρRStar)]
        
        c = max(abs(F[1]),abs(F[2]))  
        if c <= TOL
            break
        end
        @show J, U,F
        U = U - inv(J)*F
        ρRStar = U[1]
        ρLStar = U[2]
    end
    
    @show U
    ### 重新求解
    pLStar = fp(ρLStar, varL, conL,caseL)
    s_Star = fu(ρLStar, varL, conL,caseL)
    sxxLStar = Sxx(ρLStar, varL, conL)

    pRStar = fp(ρRStar, varR, conR,caseR)
    sxxRStar = Sxx(ρRStar, varR, conR)
    fL=zeros(Float64, 4)
    fR=zeros(Float64, 4)
    fL[1] = 0.0
    fL[2] = pLStar-sxxLStar
    fL[3] = (pLStar-sxxLStar)*s_Star
    fL[4] = -4conL.μ/3*s_Star
    uuh = s_Star

    fR[1] = 0.0
    fR[2] = pRStar-sxxRStar
    fR[3] = (pRStar-sxxRStar)*s_Star
    fR[4] = -4conR.μ/3*s_Star
    uuh = s_Star
    
    return fL,fR, uuh
end

A =[[1 2]; [2 3]]
typeof(A)

    ρ2=3985
    u2=0.0
    p2=1.e3
    sxx2=0
    
    
    ρ1=2785
    u1=800.0
    p1=1.e4
    sxx1=6.e4
uL = [2785, 0.0, 1.e-6,0.0]
uR = [2785, 100.0, 1.e-6,0.0]
conL = Const(3e8,2785,2.0,2.76e10,5328.0,1.338)
conR = Const(3e8,2785,2.0,2.76e10,5328.0,1.338)
fL,fR,uuh = Exact_Riemann(uL,uR,conL,conR)

J = [-2.31594 2.31594; 4.16012e7 -4.16012e7]
inv(J)

? fηη

   ρ2=2785
    u2=0.0
    p2=1.e-6
    sxx2=0
    
    
    ρ1=2785
    u1=800.0
    p1=1.e-6
    sxx1=6.e4
uL = [2785, 0.0, 1.e-6,0.0]
uR = [2785, 0.0, 1.e-6,0.0]
conL = Const(3e8,2785,2.0,2.76e10,5328.0,1.338)
conR = Const(3e8,2785,2.0,2.76e10,5328.0,1.338)
fL,fR,uuh = Exact_Riemann(uL,uR,conL,conR)
