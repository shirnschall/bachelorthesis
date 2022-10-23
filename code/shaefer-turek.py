from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
from netgen.webgui import Draw as DrawGeo

shape = Rectangle(2,.41).Circle(0.2,0.2,0.05).Reverse().Face()
shape.edges.name="wall"
shape.edges.Min(X).name="inlet"
shape.edges.Max(X).name="outlet"
shape.edges.Nearest(gp_Pnt2d(0.2,0.2)).name="cyl"

DrawGeo (shape)
mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.2))#.Curve(3)
Draw (mesh);
print (mesh.GetBoundaries())
print("mesh elements: ", mesh.ne)


def runBenchmark(Um,tend,time,drag,lift,uquer):    
    V = VectorH1(mesh,order=1, dirichlet="wall|inlet|cyl")
    Q = H1(mesh,order=1)
    X = V*Q
    

    e1 = GridFunction(X)
    e2 = GridFunction(X)
    e1.components[0].Set(CoefficientFunction((-2/(uquer**2*0.1),0)), definedon=mesh.Boundaries("cyl"))
    e2.components[0].Set(CoefficientFunction((0,-2/(uquer**2*0.1))), definedon=mesh.Boundaries("cyl"))

    u,p = X.TrialFunction()
    v,q = X.TestFunction()

    nu = 0.001

    h = specialcf.mesh_size

    stokes = (nu*InnerProduct(grad(u), grad(v))+ \
        div(u)*q+div(v)*p -0.1*h*h*grad(p)*grad(q))*dx # -1e-10*p*q)*dx #

    a = BilinearForm(stokes).Assemble()

    print("ndofs: ", X.ndof)
    # nothing here ...
    f = LinearForm(X).Assemble()

    inv_stokes = a.mat.Inverse(X.FreeDofs())
    tau = 0.001 # timestep

    m = BilinearForm(u*v*dx).Assemble()
    mstar = BilinearForm(X)
    mstar += u*v*dx+tau*stokes
    mstar.Assemble()
    invmstar = mstar.mat.Inverse(X.FreeDofs())

    conv = BilinearForm(X, nonassemble = True)
    conv += (Grad(u) * u) * v * dx
    tmpTime = []
    tmpDrag = []
    tmpLift = []
    # gridfunction for the solution
    gfu = GridFunction(X)
    #http://www.mathematik.tu-dortmund.de/lsiii/cms/papers/SchaeferTurek1996.pdf
    uin = CoefficientFunction( (Um*4*y*(0.41-y)/(0.41*0.41), 0) )
    gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
    #Draw (Norm(gfu.components[0]), mesh, "velocity", sd=3)
    #Draw (gfu.components[0], mesh, "vel");
    
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat*gfu.vec
    gfu.vec.data += inv_stokes * res

    #Draw (gfu.components[0], mesh)
    
    t = 0; i = 0
    gfut = GridFunction(V, multidim=0)
    vel = gfu.components[0]
    scene = Draw ((gfu.components[0]), mesh)

    with TaskManager():
        while t < tend:
            conv.Apply (gfu.vec, res)
            res.data += a.mat*gfu.vec
            gfu.vec.data -= tau * invmstar * res    
            
            t = t + tau; i = i + 1
            if i%10 == 0: scene.Redraw()
            if i%50 == 0: gfut.AddMultiDimComponent(vel.vec)

            tmpTime.append( t )
            tmpDrag.append(InnerProduct(res, e1.vec) )
            tmpLift.append(InnerProduct(res, e2.vec) )
            
    time.append(tmpTime)
    drag.append(tmpDrag)
    lift.append(tmpLift)

time = []
drag = []
lift = []
cl = []
ndofs = []

uquer = [0.2,1]
Um = [.3,1.5]
for i, um in enumerate(Um):
    runBenchmark(um,8,time,drag,lift,uquer[i])

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.plot(time[0], drag[0],label="Um=0.3m/s")
plt.plot(time[1], drag[1],label="Um=1.5m/s")

plt.xlabel('time')
plt.ylabel('drag')
plt.title('Drag')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(time[0], lift[0],label="Um=0.3m/s")
plt.plot(time[1], lift[1],label="Um=1.5m/s")

plt.xlabel('time')
plt.ylabel('lift')
plt.title('Lift')
plt.grid(True)
plt.legend()
plt.show()