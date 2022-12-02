import "regent"

-- run using the following command: python3 <path>/regent.py reproducer.rg -ll:gpu 1 -ll:csize 2048 -ll:fsize 2048

fspace specCord{
    x : (double[4])[2],
    y : (double[4])[2],
}

task initspecCord(coords : region(ispace(int1d), specCord)) where
writes (coords.{x, y})
do
    for coord in coords do
        for i = 0, 4 do
            coord.x[0][i] = 0.0
            coord.x[1][i] = 0.0
            coord.y[0][i] = 1.0
            coord.y[1][i] = 1.0
        end
    end
end

__demand(__cuda)
task cudaTest(coords : region(ispace(int1d), specCord)) where
reads (coords.{y}),
writes (coords.{x})
do
    for coord in coords do
        coord.x = coord.y
    end
end

task main()
    var coords = region(ispace(int1d, 16000204), specCord)
    initspecCord(coords)
    for i = 0, 1000 do
        cudaTest(coords)
    end
end

regentlib.start(main)
