def imex_solver(currentApprox, timeStep, M, mstar, invMStar, Adiff, Aconv, butcher, f=0):
    k = [currentApprox.CreateVector() for i in range(butcher.stages)]
    kHat = [currentApprox.CreateVector() for i in range(butcher.stages+1)]

    ui = currentApprox.CreateVector()
    rhsi = currentApprox.CreateVector()
    Mun = currentApprox.CreateVector()
    
    M.Apply(currentApprox, Mun)
    Aconv.Apply(currentApprox, kHat[0])
    kHat[0] *= -1

    for i in range(butcher.stages):
        rhsi[:]=0
        for j in range(i):
            if(butcher.aImp[i][j] != 0):
                rhsi.data += butcher.aImp[i][j] * k[j]
        for j in range(i+1):
            #print("explicit i= ", i,"j= ",  j)
            if (butcher.aExp[i+1][j] != 0):
                rhsi.data += butcher.aExp[i+1][j] * kHat[j]

        rhsi.data *= timeStep
        rhsi.data += Mun

        #solve MStar * ui = rhsi for ui:
        #keep nonhomogenous Dirichlet bcs
        ui.data = currentApprox
        ui.data += invMStar * (rhsi - mstar.mat*currentApprox)
        
        Adiff.Apply(ui, k[i])
        k[i] *= -1
        Aconv.Apply(ui, kHat[i+1])
        kHat[i+1] *= -1
    # return un=us since cs=1 for implicit scheme (Asher 2.3)
    return ui