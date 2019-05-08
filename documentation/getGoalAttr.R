## #' smart computation of goal angle
## #'
## #' Return the goal angle (in radians) as viewed from position (x,y),
## #' given in Opta pitch coordinates.
## #' According to Opta specs, goal ranges from coordinates (100, 45.2)
## #' to (100, 54.8). By IFAB regulations, standard pitch size is 105x68m
## #' @param x,y numeric Opta's x and y shooter coordinates
## #' @return numeric angle in radians
.goal_angle <- function(x, y) {
    if (y <= 45.2) {
        ## Angled shot from the right side of the pitch
        ## angles a + b + c = pi/2
        tb <- 0
        if (y != 45.2)
            tb <- (1.05 * (100 - x)) / (0.68 * (54.8 - y))
        if (x != 100) {
            tc <- (0.68 * (45.2 - y)) / (1.05 * (100 - x))
            return(0.5 * pi - atan(tb) - atan(tc))
        } else
            return(0)
    } else if (y < 54.8) {
        tb <- (1.05 * (100 - x)) / (0.68 * (54.8 - y))
        tc <- (1.05 * (100 - x)) / (0.68 * (y - 45.2))
        return(pi - atan(tb) - atan(tc))
    } else {
        tc <- (1.05 * (100 - x)) / (0.68 * (y - 45.2))
        if (x != 100) {
            tb <- (0.68 * (y - 54.8)) / (1.05 * (100 - x))
            return(0.5 * pi - atan(tb) - atan(tc))
        } else
            return(0)
    }
}
goal_angle <- Vectorize(.goal_angle)

.goal_distance <- function(x, y) {
    sqrt((1.05*(100 - x))^2  +  (0.68*(54.8 - y))^2)
}
goal_distance <- Vectorize(.goal_distance)

## check if free kick is dangerous: it is assumed that
## we have attacking coordinate
.isDangerousFK <- function(x, y,
                           x_min = 70, x_max = 90,
                           y_min = 10, y_max = 90) {
    (((x >= x_min) & (x <= x_max)) & ((y >= y_min) & (y <= y_max)))
}

isDangerousFK <- Vectorize(.isDangerousFK)
