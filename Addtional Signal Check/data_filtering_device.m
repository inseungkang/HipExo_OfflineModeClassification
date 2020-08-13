%%
clear all; clc; close all;

%%
filename = 'AB01_R1_L0_MECH.txt';
A = dlmread(filename,',');
%%
figure(1)
subplot(2,1,1)
plot(A(:,3))
subplot(2,1,2)
plot(A(:,4))
%% erase overshoot ticks
close all;
c = A(:,3);
d = A(:,4);
vec_1 = abs(diff(c));
vec_2 = abs(diff(d));
for i = 1:size(vec_1)
    if vec_1(i) > 5
        c(i) = nan;
    end
end
for i = 1:size(vec_2)
    if vec_2(i) > 5
        d(i) = nan;
    end
end
close all;
figure(1)
subplot(2,1,1)
plot(c)
subplot(2,1,2)
plot(d)
%% recheck overshoot
for i = 1:size(c)
    if c(i) > 70
        c(i) = nan;
    elseif c(i) < -25
            c(i) = nan;
    end
end
for i = 1:size(d)
    if d(i) > 70
        d(i) = nan;
    elseif d(i) < -25
            d(i) = nan;
    end
end
close all;
figure(1)
subplot(2,1,1)
plot(c)
subplot(2,1,2)
plot(d)
%% interp data and fill the gap
c_new = fillmissing(c,'spline');
d_new = fillmissing(d,'spline');

close all;
figure(1)
subplot(2,1,1)
plot(c_new)
subplot(2,1,2)
plot(d_new)
%%
c = A(:,3);
for i = 2:size(c)
    if abs(c(i)-c(i-1)) > 20
        c(i) = c(i-1);
    end
end

d = A(:,4);
for i = 2:size(d)
    if abs(d(i)-d(i-1)) > 20
        d(i) = d(i-1);
    end
end
%%
d = A(:,4);
for i = 2:size(d)
    if abs(d(i)-d(i-1)) > 20
        d(i) = d(i-1);
    end
end

%%
figure(1)
subplot(2,1,1)
plot(c)
subplot(2,1,2)
plot(d)
%% rewrite data
A(:,3) = c_new;
A(:,4) = d_new;
%%
csvwrite(filename,A)