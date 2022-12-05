import "regent"

local sqrt = regentlib.sqrt(double)
local pow = regentlib.pow(double)
local format = require("std/format")
local timing = require("std/timing")
local c = regentlib.c

-- runs with -ll:gpu 1 -ll:csize 5120 -ll:fsize 5120

fspace point{
    x : double,
    y : double,
    nbhs : int,
    conn : int[25],
    -- conn : regentlib.array(int, 25),
    q : regentlib.array(double, 4),
    dq : regentlib.array(double, 8),
    qm : regentlib.array(double, 8),
}

__demand(__inline)
task flatten(x : int, y : int)
    return x + 2*y
end

task init_vals(points : region(ispace(int1d), point)) where
reads writes (points) do
    var maxi = 0
    for point in points do
        point.x = 0.0001*int(point)
        point.y = 0.0001*int(point)
        point.nbhs = 11
        
        for i = 0, point.nbhs do
            point.conn[i] = (i + int(point) + 1)%16000204
            -- maxi max= point.conn[i]
        end
       
        point.q[0] = 1002.0
        point.q[1] = 120.0
        point.q[2] = 1242.0
        point.q[3] = 0.0

        for i = 0, 8 do
            point.dq[i] = 0.0
            point.qm[i] = 0.0
        end
    end
    -- format.println("maxi = {}", maxi)
end


__demand(__cuda)
task q_sim(points : region(ispace(int1d), point)) where
writes (points.{qm, dq}),
reads (points.{x, y, nbhs, conn, q, qm, dq})
do
    for i = 0, 1000 do
        for point in points do
            var x_i = point.x
            var y_i = point.y

            var sum_delx_sqr = 0.0
            var sum_dely_sqr = 0.0
            var sum_delx_dely = 0.0

            var sum_delx_delq : double[4]
            var sum_dely_delq : double[4]
            
            for l = 0, 4 do
                sum_delx_delq[l] = 0.0
                sum_dely_delq[l] = 0.0
                point.qm[flatten(0, l)] = point.q[l]
                point.qm[flatten(1, l)] = point.q[l]
            end

            for k = 0, point.nbhs do
                var nbh = point.conn[k]

                for r = 0, 4 do
                    if(points[nbh].q[r] > point.qm[flatten(0, r)]) then
                        point.qm[flatten(0, r)] = points[nbh].q[r]
                    end
                    if(points[nbh].q[r] < point.qm[flatten(1, r)]) then
                        point.qm[flatten(1, r)] = points[nbh].q[r]
                    end
                end

                var x_k = points[nbh].x
                var y_k = points[nbh].y

                var delx = x_k - x_i
                var dely = y_k - y_i

                var dist = sqrt(delx*delx + dely*dely)
                var weights = pow(dist, 2)

                var delx_weights = delx*weights
                var dely_weights = dely*weights

                sum_delx_sqr = sum_delx_sqr + delx_weights*delx
                sum_dely_sqr = sum_dely_sqr + dely_weights*dely
                sum_delx_dely = sum_delx_dely + delx_weights*dely

                var delq : double[4]

                for l = 0, 4 do
                    delq[l] = points[nbh].q[l] - point.q[l]
                    sum_delx_delq[l] += delx_weights*delq[l]
                    sum_dely_delq[l] += dely_weights*delq[l]
                end
            end
            var det = sum_delx_sqr*sum_dely_sqr - sum_delx_dely*sum_delx_dely
            var one_by_det = 1.0/det
            
            for l = 0, 4 do
                point.dq[flatten(0, l)] = one_by_det*(sum_dely_sqr*sum_delx_delq[l] - sum_delx_dely*sum_dely_delq[l])
                point.dq[flatten(1, l)] = one_by_det*(sum_delx_sqr*sum_dely_delq[l] - sum_delx_dely*sum_delx_delq[l])
            end
        end
    end
end

__demand(__cuda)
task oldCheck(points : region(ispace(int1d), point)) where
reads (points.{dq}),
writes (points.{qm})
do
    var i = 0
    while i < 1000 do
        for point in points do
            point.qm = point.dq
        end
        i+=1
    end
end

task printPoint(points : region(ispace(int1d), point)) where
reads (points.{x, y, nbhs, conn, q, qm, dq})
do
    for point in points do
        format.println("point = {}, x = {}, y = {}, nbhs = {}", int(point), point.x, point.y, point.nbhs)
        for i = 0, point.nbhs do
            format.println("conn[{}] = {}", i, point.conn[i])
        end
        for i = 0, 4 do
            format.println("q[{}] = {}", i, point.q[i])
        end
        for i = 0, 2 do
            for j = 0, 4 do
                format.println("qm[{}, {}] = {}", i, j, point.qm[flatten(i, j)])
            end
        end
        for i = 0, 2 do
            for j = 0, 4 do
                format.println("dq[{}, {}] = {}", i, j, point.dq[flatten(i, j)])
            end
        end
    end
end

task main()
    var points = region(ispace(int1d, 16000204), point)
    init_vals(points)
    __fence(__execution, __block)
    var itime = regentlib.c.legion_get_current_time_in_micros()
    q_sim(points)
    __fence(__execution, __block)
    var ftime = regentlib.c.legion_get_current_time_in_micros()
    format.println("Time taken: {} seconds", double(ftime - itime)/1e6)
end

regentlib.start(main)

