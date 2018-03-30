import sys
import tester
import regressor
import basis

if __name__=='__main__':
    if len(sys.argv) < 2:
        print ( 'Choose one of the basis function' )
        print ( '  - Simple' )
        print ( '  - Power' )
        print ( '  - Gaussisn (optional<sigma>(default=1))' )
        print ( '  - Sigmoid (optional<sigma>(default=1))' )
        print ( 'example : " python main.py Gaussian 0.02 "' )
        exit(1)
    elif sys.argv[1] == 'Simple':
        b = basis.SimpleBasis()
    elif sys.argv[1] == 'Power':
        b = basis.PowerBasis()
    elif sys.argv[1] == 'Gaussian':
        if len(sys.argv) > 2:
            b = basis.GaussianBasis(float(sys.argv[2]))
        else:
            b = basis.GaussianBasis()
    elif sys.argv[1] == 'Sigmoid':
        if len(sys.argv) > 2:
            b = basis.SigmoidBasis(float(sys.argv[2]))
        else:
            b = basis.SigmoidBasis()
    else:
        print ('invalid basis function')
        exit(1)

    r = regressor.LinearRegressor(b)
    ccpp = tester.CCPPTester(r)
    ccpp.run()
