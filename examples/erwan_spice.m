%%
clear

%% initialize SPICE

PATH_MICE='/Users/emazaric/Work/SPICE/mice';
addpath([PATH_MICE,'/src/mice/']);
addpath([PATH_MICE,'/lib/']);

clktol='10:000';

FURNSHtime='simple.furnsh';
cspice_kclear;
cspice_furnsh(FURNSHtime);

%% Define time window

utc0='2011 MAR 01 00:00:00.00';
utc1='2012 MAR 01 00:00:00.00';
stepet=3600;

et0=cspice_str2et(utc0);
et1=cspice_str2et(utc1);
et_=et0:stepet:et1;
nbet=length(et_)

%% Sun positions over time period

possun=cspice_spkpos('SUN',et_,'MOON_ME','LT+S','MOON')';
[lonsun,latsun,radsun]=cart2sph(possun(:,1),possun(:,2),possun(:,3));
lonsun=mod(lonsun*180/pi,360);
latsun=latsun*180/pi;

%% read GRD
prec='double';

ncid = netcdf.open(GRDfile, 'NC_NOWRITE');
[ndims,nvars,ngatts,unlimdimid] = netcdf.inq(ncid);

eval(['grdx=',prec,'(netcdf.getVar(ncid,0));']);
eval(['grdy=',prec,'(netcdf.getVar(ncid,1));']);
nx=length(grdx);
ny=length(grdy);

grdz_=zeros(nx,ny,prec);
grdz_(:)=netcdf.getVar(ncid,2);
grdz_=grdz_';

netcdf.close(ncid);

[grdx_,grdy_]=meshgrid(grdx,grdy);

%% unprotect from stereographic to lon/lat
lon0=0; lat0=-90; R=1737.4;

rho=sqrt(grdx_.^2+grdy_.^2);
c=2*atan2(rho,2*R);

lat_=180/pi*asin(cos(c).*sind(lat0)+(cosd(lat0).*y.*sin(c))./rho);
lon_=mod(lon0+180/pi*atan2(x.*sin(c),cosd(lat0).*rho.*cos(c)-sind(lat0).*y.*sin(c)),360);

lat(and(grdx_==0,grdy_==0))=lat0;
lon(and(grdx_==0,grdy_==0))=lon0;

%% go from lon/lat to cartesian

[xc_,yc_,zc_]=sph2cart(lon_/180*pi,lat_/180*pi,R+grdz_);