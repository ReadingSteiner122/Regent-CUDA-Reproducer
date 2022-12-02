import "regent"

local format = require("std/format")

-- run using the following command: python3 <path>/regent.py reproducer.rg -ll:gpu 1 -ll:csize 2048 -ll:fsize 2048

fspace specCord{
    -- x : (double[4])[2],
    -- y : (double[4])[2],
    x : regentlib.array(double, 8),
    y : regentlib.array(double, 8),
}

__demand(__inline)
task c(x : int, y : int)
  return x + 2*y
end

task initspecCord(coords : region(ispace(int1d), specCord)) where
writes (coords.{x, y})
do
    for coord in coords do
        for i = 0, 4 do
            -- coord.x[0][i] = 0.0
            -- coord.x[1][i] = 0.0
            -- coord.y[0][i] = 1.0
            -- coord.y[1][i] = 1.0
            coord.x[c(0, i)] = 0.0
            coord.x[c(1, i)] = 0.0
            coord.y[c(0, i)] = 1.0
            coord.y[c(1, i)] = 1.0
        end
    end
end

__demand(__cuda)
task cudaTest(coords : region(ispace(int1d), specCord)) where
reads (coords.{y}),
writes (coords.{x})
do
    var t = 0
    while t < 1000 do
        for coord in coords do
            coord.x = coord.y
        end
        t += 1
    end
end

task main()
    var coords = region(ispace(int1d, 16000204), specCord)
    initspecCord(coords)
    __fence(__execution, __block)
    var start = regentlib.c.legion_get_current_time_in_micros()
    -- for i = 0, 1000 do
    --     cudaTest(coords)
    -- end
    cudaTest(coords)
    __fence(__execution, __block)
    var stop = regentlib.c.legion_get_current_time_in_micros()
    format.println("Time taken by function: {}", double(stop-start)/1e6)
end

regentlib.start(main)
